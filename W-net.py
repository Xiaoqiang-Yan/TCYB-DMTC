# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import shutil

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import argparse
import os
import time
import itertools
import logging
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from keras.preprocessing.image import ImageDataGenerator
import yaml
from models.util_ex import load_model

import models
from utils.util import create_logger, AverageMeter, clustering_acc, save_checkpoint, load_checkpoint, save_list, load_list, process_result, show_result
from utils.sampling import get_pair_EX
from utils.mc_dataset import McDataset
from utils.function_alexnet import get_dim, forward, comp_simi

# argparser
parser = argparse.ArgumentParser(description='PyTorch Implementation of DCCM')
parser.add_argument('--resume', default=None, type=str, help='resume from a checkpoint')
parser.add_argument('--config', default='cfgs/config.yaml', help='set configuration file')
parser.add_argument('--small_bs', default=32, type=int)
parser.add_argument('--order', default=0, type=int)
parser.add_argument('--input_size', default=96, type=int)
parser.add_argument('--split', default=None, type=int, help='divide the large forward batch to avoid OOM')

args = parser.parse_args()

with open(args.config) as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
for k, v in config['common'].items():
	setattr(args, k, v)
coeff = config['coeff']
start_epoch = 0

def main():
	global args,  start_epoch
	task_num=3
	logger = create_logger('global_logger', log_file=os.path.join(args.save_path,'log.txt'))
	print("=============="+" Training for Task "+str(args.order)+" ==============")

	model = models.__dict__[args.arch](num_classes = args.num_classes).cuda()

	abs_path = os.path.abspath('.')
	model_args = abs_path + "/checkpoint/Cifar-100/task_all_Cifar_100/ckpt.pth.tar"
	checkpoint = torch.load(model_args)
	pretrained_dict = checkpoint["model"]
	model_param = ['top_layer.0.weight', 'top_layer.0.bias', 'top_layer.1.weight', 'top_layer.1.bias']
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in model_param}
	model.load_state_dict(pretrained_dict)

	toy_input = torch.zeros([5, 3, args.input_size, args.input_size]).cuda()
	arch_info = get_dim(model, toy_input, args.layers, args.c_layer)
	dim_loss = models.__dict__['DIM_Loss'](arch_info).cuda()

	para_dict = itertools.chain(filter(lambda x: x.requires_grad, model.parameters()),
		  filter(lambda x: x.requires_grad, dim_loss.parameters()))
	optimizer = torch.optim.RMSprop(para_dict, lr=args.lr, alpha=0.9)
	best_acc = 0; c_nmi = 0
	crit_graph = nn.BCELoss().cuda()

	if args.resume:
		logger.info("=> loading checkpoint '{}'".format(args.resume))
		start_epoch, best_nmi = load_checkpoint(model, dim_loss, optimizer, args.resume)

	tra = [transforms.Resize(224),
		   transforms.CenterCrop(224),
		   transforms.ToTensor()]
	dataset = McDataset(
		  args.root,
		  args.source,
		  transform=transforms.Compose(tra))
	dataloader = torch.utils.data.DataLoader(
		  dataset, batch_size=args.large_bs,
		  num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)
	datagen = ImageDataGenerator(
		  rotation_range=20,
		  width_shift_range=0.18,
		  height_shift_range=0.18,
		  channel_shift_range=0.1,
		  horizontal_flip=True,
		  rescale=0.95,
		  zoom_range=[0.85,1.15])

	
	for epoch in range(start_epoch, args.epochs):

		end = time.time()

		nmi, acc = test(dataloader, model, epoch)
		is_best_acc = acc > best_acc
		best_acc = max(acc, best_acc)
		if(acc == best_acc):
			c_nmi = nmi

		save_checkpoint({
			  'epoch': epoch, 
			  'model': model.state_dict(), 
			  'dim_loss': dim_loss.state_dict(), 
			  'best_acc': best_acc,
			  'optimizer': optimizer.state_dict()}, 
			  is_best_acc, args.save_path + '/ckpt')

		print("iteration: " + str(epoch + 1))
		train(dataloader, model, dim_loss, crit_graph, optimizer, epoch, datagen)

	rm_path = args.save_path + "/ckpt.pth.tar"
	rm_path_2 = args.save_path + "/log.txt"
	os.remove(rm_path)
	os.remove(rm_path_2)

	if (os.path.exists("C0_result.npy")):
		task_n = load_list("C0_result.npy")
	else:
		task_n = []
	task_n.append([best_acc, c_nmi])
	save_list(task_n, "C0_result.npy")

	if (os.path.exists("C_result.npy")):
		task_result = load_list("C_result.npy")
	else:
		task_result = []
	task_n = load_list("C0_result.npy")
	task_result.append(task_n)
	save_list(task_result,"C_result.npy")
	os.remove("C0_result.npy")

	if (args.order >=task_num):
		task_final = load_list("C_result.npy")
		show_result(task_final)
		c_r = os.path.abspath('.') + "/data/C_result.npy"
		if (os.path.exists(c_r)):
			os.remove(c_r)
			shutil.copy("C_result.npy", c_r)
			os.remove("C_result.npy")
		else:
			shutil.copy("C_result.npy", c_r)
			os.remove("C_result.npy")


def train(loader, model, dim_loss, crit_graph, optimizer, epoch, datagen):
		
	freq = args.print_freq

	batch_time = AverageMeter(freq)
	data_time = AverageMeter(freq)
	losses = AverageMeter(freq)
	g_losses = AverageMeter(freq)
	loc_losses = AverageMeter(freq)
	
	logger = logging.getLogger('global_logger')

	model.train()
	dim_loss.train()

	index_loc = np.arange(args.large_bs)
	end = time.time()

	for i, (input_tensor, target) in enumerate(loader):
		data_time.update(time.time() - end)
		input_var = torch.autograd.Variable(input_tensor.cuda())
		target = target.cuda()

		with torch.no_grad():
			if args.split:
				vec_list = []
				bs = args.large_bs // args.split
				for kk in range(args.split):
					temp, _, _ = forward(model, input_var[kk*bs:(kk+1)*bs], 
						  args.layers, args.c_layer)
					vec_list.append(temp)
				vec = torch.cat(vec_list, dim=0)
			else:
				vec, _, _ = forward(model, input_var, args.layers, args.c_layer)

		similarity, labels, weights = comp_simi(vec)
		mask = similarity.ge(args.thresh)

		for k in range(args.repeated):
			np.random.shuffle(index_loc)
			for j in range(similarity.shape[0] // args.small_bs):
				address = index_loc[np.arange(j*args.small_bs,(j+1)*args.small_bs)]
				input_bs = input_tensor[address]
				gt_target = target[address]
				input_bs = input_bs.numpy()
				out_target = labels[address]
				out_target = out_target.detach()
				mask_target = mask[address,:][:,address].float()
				weights_batch = weights[address]

				sign = 0
				for X_batch_i in datagen.flow(input_bs,batch_size=args.small_bs,shuffle=False):
					aug_input_bs = torch.from_numpy(X_batch_i)
					aug_input_bs = aug_input_bs.float()
					aug_input_batch_var = torch.autograd.Variable(aug_input_bs.cuda())
					vec, [M,Y], c_vec = forward(model, aug_input_batch_var, 
						  args.layers, args.c_layer)

					simi_batch, labels_batch, weigths_tmp = comp_simi(vec)
					simi_batch = simi_batch/torch.max(simi_batch)

					Y_aug, M = get_pair_EX(Y, M, mask_target)
					_local = dim_loss(Y_aug, M)
					_graph = crit_graph(simi_batch, mask_target)

					loss = 1.0 * _graph + 0.5 * _local

					losses.update(loss.item())
					g_losses.update(_graph.item())
					loc_losses.update(_local.item())

					optimizer.zero_grad()
					loss.backward(retain_graph=True)
					optimizer.step()
		
					sign += 1
					if sign > 1:
						break

					batch_time.update(time.time() - end)
					end = time.time()

def test(loader, model, epoch):
	logger = logging.getLogger('global_logger')
	model.eval()
	gnd_labels = []
	pred_labels = []
	for i, (input_tensor, target) in enumerate(loader):
		input_var = torch.autograd.Variable(input_tensor.cuda())
		with torch.no_grad():
			if args.split:
				vec_list = []
				bs = args.large_bs // args.split
				for kk in range(args.split):
					temp, _, _ = forward(model, input_var[kk*bs:(kk+1)*bs], 
						  args.layers, args.c_layer)
					vec_list.append(temp)
				vec = torch.cat(vec_list, dim=0)
			else:
				vec, _, _ = forward(model, input_var, args.layers, args.c_layer)

		_, indices = torch.max(vec, 1)
		gnd_labels.extend(target.data.numpy())
		pred_labels.extend(indices.data.cpu().numpy())

	gnd_labels = np.array(gnd_labels)
	pred_labels = np.array(pred_labels)
	
	nmi = normalized_mutual_info_score(gnd_labels, pred_labels)
	acc = clustering_acc(gnd_labels, pred_labels)

	return nmi, acc

if __name__ == '__main__':
	main()
# -*- coding: utf-8 -*-
import torch
import numpy as np
import random

import argparse
import os
import pickle
import time
import itertools
import pdb
import logging
from tensorboardX import SummaryWriter
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from keras.preprocessing.image import ImageDataGenerator
import yaml
from models.util_ex import load_model

import models
from utils.util import create_logger, AverageMeter, Logger, clustering_acc, WeightedBCE, accuracy, save_checkpoint, load_checkpoint, save_checkpoint_ex, download_model
from utils.sampling import get_pair, get_pair_EX
from utils.mc_dataset import McDataset
from utils.function_alexnet import get_dim, forward, comp_simi

# argparser
parser = argparse.ArgumentParser(description='PyTorch Implementation of DCCM')
parser.add_argument('--resume', default=None, type=str, help='resume from a checkpoint')
parser.add_argument('--config', default='cfgs/config.yaml', help='set configuration file')
parser.add_argument('--small_bs', default=32, type=int)
parser.add_argument('--input_size', default=96, type=int)
parser.add_argument('--split', default=None, type=int, help='divide the large forward batch to avoid OOM')

args = parser.parse_args()

with open(args.config) as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
for k, v in config['common'].items():
	setattr(args, k, v)
coeff = config['coeff']

best_nmi = 0
start_epoch = 0

def main():
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        np.random.seed(1234)
        random.seed(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
	global args, best_nmi, start_epoch
	logger = create_logger('global_logger', log_file=os.path.join(args.save_path,'log.txt'))
	download_model()
	print("======================" + " Training for auxiliary clustering " + " ======================")

	model = models.__dict__[args.arch](num_classes = args.num_classes).cuda()
	abs_path = os.path.abspath('.')
	model_args = abs_path + "/models/parameter/checkpoint.pth.tar"
	alexnet_pre = load_model(model_args)
	pretrained_dict = alexnet_pre.state_dict()
	model_dict = model.state_dict()
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)

	toy_input = torch.zeros([5, 3, args.input_size, args.input_size]).cuda()
	arch_info = get_dim(model, toy_input, args.layers, args.c_layer)
	dim_loss = models.__dict__['DIM_Loss'](arch_info).cuda()

	para_dict = itertools.chain(filter(lambda x: x.requires_grad, model.parameters()),
		  filter(lambda x: x.requires_grad, dim_loss.parameters()))
	optimizer = torch.optim.RMSprop(para_dict, lr=args.lr, alpha=0.9)

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
		test(dataloader, model, epoch)
		save_checkpoint_ex({
			  'epoch': epoch, 
			  'model': model.state_dict(), 
			  'dim_loss': dim_loss.state_dict(),
			  'optimizer': optimizer.state_dict()}, 
			   args.save_path + '/ckpt' + str(epoch))

		# training
		train(dataloader, model, dim_loss, crit_graph, optimizer, epoch, datagen)
	rm_path_2 = args.save_path + "/log.txt"
	os.remove(rm_path_2)


def train(loader, model, dim_loss, crit_graph, optimizer, epoch, datagen):
		
	freq = args.print_freq

	batch_time = AverageMeter(freq)
	data_time = AverageMeter(freq)
	losses = AverageMeter(freq)
	g_losses = AverageMeter(freq)
	l_losses = AverageMeter(freq)
	loc_losses = AverageMeter(freq)
	
	logger = logging.getLogger('global_logger')

	# switch to train mode
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

					# records
					losses.update(loss.item())
					g_losses.update(_graph.item())
					loc_losses.update(_local.item())

					optimizer.zero_grad()
					loss.backward(retain_graph=True)
					optimizer.step()
		
					sign += 1
					if sign > 1:
						break

					# measure elapsed time
					batch_time.update(time.time() - end)
					end = time.time()

		if i % args.print_freq == 0:	
			step = epoch * len(loader) + i
			print('Epoch: [{0}/{1}][{2}/{3}]\t'
				  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
					epoch, args.epochs, i, len(loader), 
					batch_time=batch_time,
					data_time=data_time,
					))


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
					temp, _, _ = forward(model, input_var[kk * bs:(kk + 1) * bs],
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


if __name__ == '__main__':
	main()

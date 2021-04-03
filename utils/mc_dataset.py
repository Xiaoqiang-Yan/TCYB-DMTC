from torch.utils.data import DataLoader, Dataset
import numpy as np
from skimage import io, color
from PIL import Image
import torchvision.transforms as transforms
import torch
from skimage.transform import resize
import pdb

class McDataset(Dataset):
	def __init__(self, root_dir, meta_file, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		with open(meta_file) as f:
			lines = f.readlines()
		self.num = len(lines)
		self.metas = []
		self.imgs = []
		for line in lines:
			path, cls = line.rstrip().split()
			self.metas.append((path, int(cls)))
			self.imgs.append((self.root_dir + '/' + path, int(cls)))
 
	def __len__(self):
		return self.num

	def __getitem__(self, idx):
		filename = self.root_dir + '/' + self.metas[idx][0]
		cls = self.metas[idx][1]
		img = Image.open(filename)
		img = img.convert('RGB')
		if self.transform:
			img = self.transform(img)
		return img, cls


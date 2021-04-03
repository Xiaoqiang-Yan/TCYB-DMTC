# -*- coding: utf-8 -*-
import torch
import pdb

def get_fake_M(M, mask):
	M_fake = []
	for i in range(len(M)):
		count = 0
		for j in range(i, 2*len(M)):
			if count > 1:
				break
			if j > len(M) -1:
				j = j % len(M)
				if mask[i][j] == 1 or i == j:
					continue
				else:
					M_fake.append(M[j].unsqueeze(0))
					count += 1
	if len(M_fake) < 2 * M.size(0):
		raise Exception('threshold is too low')
	return torch.cat((M_fake[::2]+ M_fake[1::2]), dim=0)


def get_pos_M(M, mask):
	M_pos = []
	for i in range(M.size(0)):
		indice = torch.topk(torch.cat([mask[i][:i],mask[i][i+1:]]),1)[1].item()
		M_pos.append(M[indice].unsqueeze(0))

	return torch.cat(M_pos, dim=0)

def get_pair(Y, M, mask):
	M_fake = get_fake_M(M, mask)
	M_pos = get_pos_M(M, mask)
	M = torch.cat([M, M_pos], dim=0)
	Y_aug = torch.cat([Y, Y], dim=0)

	return Y_aug, M, M_fake

def get_pair_EX(Y, M, mask):
	M_pos1, M_pos2 = get_pos_M_EX(M, mask)
	M = torch.cat([M_pos1, M_pos2], dim=0)
	Y_aug = torch.cat([Y, Y], dim=0)

	return Y_aug, M


def get_pos_M_EX(M, mask):
	M_pos_1 = []
	M_pos_2 = []
	for i in range(M.size(0)):
		indice = torch.topk(torch.cat([mask[i][:i],mask[i][i+1:]]),2)
		indice1 = indice[1][0].item()
		indice2 = indice[1][1].item()
		M_pos_1.append(M[indice1].unsqueeze(0))
		M_pos_2.append(M[indice2].unsqueeze(0))

	return torch.cat(M_pos_1, dim=0), torch.cat(M_pos_2, dim=0)


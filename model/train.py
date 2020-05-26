#!/usr/bin/python3

import torch
from torch.utils import data
from data_generator import Dataset

def main():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Parameters
	image_size = 299
	params = {'batch_size': 64,
	          'shuffle': True,
	          'num_workers': 6}



	training_set = Dataset( '/home/jasondent/art_telephone/model/data/train' , ['pics','wimmel'], image_size )
	training_generator = data.DataLoader(training_set, **params)

	training_set = Dataset( '/home/jasondent/art_telephone/model/data/test' , ['pics','wimmel'], image_size )
	training_generator = data.DataLoader(training_set, **params)

	for local_batch, local_labels in training_generator:
	    # Transfer to GPU
	    print()
		local_batch, local_labels = local_batch.to(device), local_labels.to(device)
		print(local_batch.shape)


if __name__ == '__main__':
	main()
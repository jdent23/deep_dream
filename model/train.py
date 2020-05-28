#!/usr/bin/python3

import torch
from torch.utils import data
from data_generator import Dataset
from model import Classifier

def main():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = Classifier().to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
	f_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

	# Parameters
	image_size = 299
	params = {'batch_size': 16,
		'shuffle': True,
		'num_workers': 6}



	training_set = Dataset( '/home/jasondent/art_telephone/model/data/train' , ['pics','wimmel'], image_size )
	training_generator = data.DataLoader(training_set, **params)

	testing_set = Dataset( '/home/jasondent/art_telephone/model/data/test' , ['pics','wimmel'], image_size )
	testing_generator = data.DataLoader(training_set, **params)

	for epoch in range( 0, 20 ):
		for batch, labels in training_generator:
			y = labels[ :, 0 ].reshape( ( -1, 1 ) )
			batch, y = batch.to(device), y.to(device)

			p_y = model.induction( batch, train=True )


			optimizer.zero_grad()
			loss = f_loss( y, p_y )
			loss.backward()
			optimizer.step()

		for batch, labels in testing_generator:
			y = labels[ :, 0 ].reshape( ( -1, 1 ) )
			batch, y = batch.to(device), y.to(device)

			p_y = model.induction( batch, train=True )
			loss = f_loss( y, p_y )
			print( p_y )


if __name__ == '__main__':
	main()
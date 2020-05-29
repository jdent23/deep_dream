#!/usr/bin/python3

import torch
from torch.utils import data
from data_generator import Dataset
from model import Classifier
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

def main():

	writer = SummaryWriter()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = Classifier().to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
	f_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

	# Parameters
	image_size = 299
	params = {'batch_size': 10,
		'shuffle': True,
		'num_workers': 6}

	data_transform = transforms.Compose([\
		transforms.RandomHorizontalFlip(),\
		transforms.RandomVerticalFlip(),\
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)\
	])



	training_set = Dataset( '/home/jasondent/art_telephone/model/data/train' , ['pics','wimmel'], image_size )
	training_generator = data.DataLoader(training_set, **params)

	testing_set = Dataset( '/home/jasondent/art_telephone/model/data/test' , ['pics','wimmel'], image_size )
	testing_generator = data.DataLoader(training_set, **params)

	n_iter = -1
	for epoch in range( 0, 20 ):
		for batch, labels in training_generator:
			n_iter += 1
			y = labels[ :, 0 ].reshape( ( -1, 1 ) )
			batch, y = batch.to(device), y.to(device)

			p_y = model.induction( batch, train=True )


			optimizer.zero_grad()
			loss = f_loss( y, p_y )
			loss.backward()
			optimizer.step()

		total_loss = 0
		for batch, labels in testing_generator:
			y = labels[ :, 0 ].reshape( ( -1, 1 ) )
			batch, y = batch.to(device), y.to(device)

			p_y = model.induction( batch, train=True )
			loss = f_loss( y, p_y )
			total_loss += loss

		writer.add_scalar('test/loss', np.random.random(), n_iter)
		print( total_loss )


if __name__ == '__main__':
	main()

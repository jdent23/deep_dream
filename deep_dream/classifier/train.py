#!/usr/bin/python3

import torch
from torch.utils import data
from deep_dream.classifier.data_generator import Dataset
from deep_dream.classifier.model import Classifier
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

def main():

	writer = SummaryWriter()

	device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )

	allowable_bad_steps = 10
	PATH = './classifier.pt'
	model = Classifier().to( device )

	try:
		model.load_state_dict( torch.load( PATH ) )
		print( 'Loading existing model' )
	except:
		print( 'Model not found; starting new model' )

	optimizer = torch.optim.SGD( model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4 )
	f_loss = torch.nn.MSELoss( size_average=None, reduce=None, reduction='mean' )

	# Parameters
	image_size = 299
	params = {'batch_size': 32,
		'shuffle': True,
		'num_workers': 6}

	data_transform = transforms.Compose([\
		transforms.RandomHorizontalFlip(),\
		transforms.RandomVerticalFlip(),\
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)\
	])
	
	imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
	data_loader = torch.utils.data.DataLoader(
		imagenet_data,
		batch_size=4,
		shuffle=True,
		num_workers=6
	)

	testing_set = Dataset( '/home/jasondent/art_telephone/model/data/test' , ['pics','wimmel'], image_size )
	testing_generator = data.DataLoader(training_set, **params)


	lowest_total_loss = -1
	steps_since_lowest_loss = 0
	n_iter = -1
	while( True ):
		for batch, labels in training_generator:
			n_iter += 1
			y = labels[ :, 0 ].reshape( ( -1, 1 ) )
			del labels
			batch, y = batch.to(device), y.to(device)

			model.train()
			p_y = model( batch, train=True )
			del batch


			optimizer.zero_grad()
			loss = f_loss( y, p_y )
			writer.add_scalar('train/loss', loss, n_iter)
			loss.backward()
			optimizer.step()
			del loss

		total_loss = 0
		for batch, labels in testing_generator:
			y = labels[ :, 0 ].reshape( ( -1, 1 ) )
			del labels
			batch, y = batch.to(device), y.to(device)

			model.eval()
			p_y = model( batch, train=True )
			del batch
			print(y)
			print(p_y)
			loss = f_loss( y, p_y )
			total_loss += loss.detach().cpu().item()
			del loss

		writer.add_scalar('test/loss', total_loss, n_iter)
		if( ( lowest_total_loss == -1 ) | ( total_loss < lowest_total_loss ) ):
			print( 'Saving updated model; total_loss = ' + str( total_loss ) )
			torch.save(model.state_dict(), PATH)
			steps_since_lowest_loss = 0
			lowest_total_loss = total_loss
		else:
			print( 'Model not improved; total loss = ' + str( total_loss ) )
			steps_since_lowest_loss += 1

		if( steps_since_lowest_loss > allowable_bad_steps ):
			print( 'Number of allowable bad steps exceded; total loss of final model: ' + str( lowest_total_loss ) )
			break


if __name__ == '__main__':
	main()

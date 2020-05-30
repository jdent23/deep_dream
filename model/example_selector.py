#!/usr/bin/python3

import torch
from torch.utils import data
from data_generator import Dataset
from model import Classifier
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
import cv2
import hashlib

def main():

	device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )

	PATH = './class_examples/'
	model = Classifier().to( device )
	model.eval()

	try:
		model.load_state_dict( torch.load( './classifier.pt' ) )
		print( 'Loading existing model' )
	except:
		raise Exception( 'Model not found' )

	f_loss = torch.nn.MSELoss( size_average=None, reduce=None, reduction='mean' )

	# Parameters
	image_size = 299
	params = {'batch_size': 1,
		'shuffle': True,
		'num_workers': 6}

	dataset = Dataset( '/home/jasondent/art_telephone/model/data/train' , ['pics','wimmel'], image_size )
	data_generator = data.DataLoader(dataset, **params)

	resume = True
	while( resume ):
		for batch, labels in data_generator:
			del labels
			batch = batch.to(device)
			p_y = model( batch, train=True )[ 0 ][ 0 ].detach().cpu().item()
			print(p_y)
			if( p_y < 0.1):
				np_batch = batch[ 0 ].permute( 1, 2, 0 ).cpu().numpy()
				
				cv2.imshow('image',np_batch)
				k = cv2.waitKey(0) & 0xFF
				if( k == 113 ):
					print( 'quitting_program' )
					resume = False
					cv2.destroyAllWindows()
					break
				elif( True ):
					print( 'Copying image to examples' )
					np_batch *= 255
					cv2.imwrite( PATH + str( hashlib.sha224( np_batch ).hexdigest() ) + '.png' ,np_batch )
				else:
					print( 'Continuing to next example' )

				cv2.destroyAllWindows()
			del batch
			del p_y


if __name__ == '__main__':
	main()

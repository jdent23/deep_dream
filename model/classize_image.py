#!/usr/bin/python3

import torch
from torch.utils import data
from data_generator import Dataset
from model import Classifier
import cv2
import argparse
import random
import math

def show_tensor( input_tensor ):
	cv2.imshow('image', input_tensor.permute( 1, 2, 0 ).cpu().detach().numpy() )
	cv2.waitKey(0) & 0xFF
	cv2.destroyAllWindows()

def classize_image( input_tensor, example_tensor, model, iterations, blocks_deep=None ):

	input_tensor = input_tensor.repeat( example_tensor.shape[0], 1, 1, 1 )

	optimizer = torch.optim.SGD( model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4 )
	f_loss = torch.nn.MSELoss( size_average=None, reduce=None, reduction='mean' )

	x = model( example_tensor, blocks_deep=blocks_deep )

	input_tensor = torch.nn.functional.pad(\
		input_tensor,\
		( model.image_size-1, model.image_size-1, model.image_size-1, model.image_size-1 ),\
		mode='constant',\
		value=0\
	)

	num_tiles = ( ( math.ceil( input_tensor.shape[ 2 ] / model.image_size )-1 ),\
				( math.ceil( input_tensor.shape[ 3 ] / model.image_size )-1 ) )
	jitter = ( math.floor( model.image_size * random.random() ),\
				math.floor( model.image_size * random.random() ) )

	for x_tile in range( 0, num_tiles[ 0 ] ):
		for y_tile in range( 0, num_tiles[ 1 ] ):
					tile = input_tensor[\
						:,\
						:,\
						x_tile*model.image_size+jitter[0]:(x_tile+1)*model.image_size+jitter[0],\
						y_tile*model.image_size+jitter[1]:(y_tile+1)*model.image_size+jitter[1]\
					]
					print(tile.shape)
					print(input_tensor.shape)
					print((x_tile+1)*model.image_size+jitter[0])
					print((y_tile+1)*model.image_size+jitter[1])

def main( args ):

	if( args.image == None ):
		raise Exception( '--image argument required' )
	if( args.iterations == None ):
		args.iterations = 1

	device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
	model = Classifier().to( device )

	try:
		model.load_state_dict( torch.load( './classifier.pt' ) )
		print( 'Loading existing model' )
	except:
		raise Exception( 'Model not found' )
	model.eval()

	params = {'batch_size': 4,
		'shuffle': True,
		'num_workers': 6}

	dataset = Dataset( '/home/jasondent/art_telephone/model/' , ['class_examples'], 299 )
	data_generator = data.DataLoader(dataset, **params)

	img = cv2.imread( args.image )
	img_tensor = torch.from_numpy( img ).to( device ).permute( 2, 0, 1 )[ None, :, :, : ].type( torch.FloatTensor ) / 255

	for batch, labels in data_generator:
		del labels
		batch = batch.to( device )
		classize_image( img_tensor, batch, model, args.iterations )


if __name__ == '__main__':

	parser = argparse.ArgumentParser( description='Transform an input image to look more like an example image' )
	parser.add_argument( '--image', type=str, help='a url pointing to a png image bigger than 299 x 299 pixels' )
	parser.add_argument( '--iterations', type=str, help='number of times to run the image through processing' )
	args = parser.parse_args()

	main( args )

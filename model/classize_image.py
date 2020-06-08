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

def classize_image( input_tensor_img, example_tensor, model, iterations, blocks_deep=None ):

	device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
	input_tensor = input_tensor_img.repeat( example_tensor.shape[0], 1, 1, 1 ).to( device )

	f_loss = torch.nn.MSELoss( size_average=None, reduce=None, reduction='mean' )

	input_tensor = torch.nn.functional.pad(\
		input_tensor,\
		( model.image_size-1, model.image_size-1, model.image_size-1, model.image_size-1 ),\
		mode='constant',\
		value=0\
	)



	for iter in range( 0, iterations ):

		jitter = ( math.floor( model.image_size * random.random() ),\
			math.floor( model.image_size * random.random() ) )
		num_tiles = ( ( math.ceil( ( input_tensor.shape[ 2 ] - jitter[ 0 ] ) / model.image_size )-1 ),\
			( ( math.ceil( ( input_tensor.shape[ 3 ] - jitter[ 1 ] ) / model.image_size )-1 ) ) )
		for x_tile in range( 0, num_tiles[ 0 ] ):
			for y_tile in range( 0, num_tiles[ 1 ] ):
				tile = input_tensor[\
					:,\
					:,\
					x_tile*model.image_size+jitter[0]:(x_tile+1)*model.image_size+jitter[0],\
					y_tile*model.image_size+jitter[1]:(y_tile+1)*model.image_size+jitter[1]\
				].requires_grad_(True)
				optimizer = torch.optim.SGD( [ tile ], lr=10, momentum=0.9, weight_decay=1e-4 )
				original_tile = tile

				if( ( tile.shape[ 2 ] != 299 ) | ( tile.shape[ 3 ] != 299 ) ):
					print(tile.shape)
					print(x_tile)
					print(y_tile)
					print((x_tile+1)*model.image_size+jitter[0])
					print((y_tile+1)*model.image_size+jitter[1])
					show_tensor( tile[0] )

				target_output = model( example_tensor, train=True, blocks_deep=blocks_deep )
				real_output = model( tile, train=True, blocks_deep=blocks_deep )

				optimizer.zero_grad()
				loss = f_loss( target_output, real_output )
				#loss = torch.sum( torch.abs( tile - example_tensor ) )
				loss.backward()
				optimizer.step()

				input_tensor[\
					:,\
					:,\
					x_tile*model.image_size+jitter[0]:(x_tile+1)*model.image_size+jitter[0],\
					y_tile*model.image_size+jitter[1]:(y_tile+1)*model.image_size+jitter[1]\
				] = tile.detach()

	show_tensor( input_tensor.mean( 0 ) )

def main( args ):

	if( args.image == None ):
		raise Exception( '--image argument required' )
	if( args.iterations == None ):
		args.iterations = 1000

	device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
	model = Classifier()

	try:
		model.load_state_dict( torch.load( './classifier.pt' ) )
		print( 'Loading existing model' )
	except:
		raise Exception( 'Model not found' )
	model = model.to( device )

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
		classize_image( img_tensor, batch, model, args.iterations, blocks_deep=6 )


if __name__ == '__main__':

	parser = argparse.ArgumentParser( description='Transform an input image to look more like an example image' )
	parser.add_argument( '--image', type=str, help='a url pointing to a png image bigger than 299 x 299 pixels' )
	parser.add_argument( '--iterations', type=str, help='number of times to run the image through processing' )
	args = parser.parse_args()

	main( args )

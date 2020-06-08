#!/usr/bin/python3

import torch
from torch.utils import data
from data_generator import Dataset
from model import Classifier
import cv2
import argparse
import random
import math
import tensor_utils as utils



def classize_image( input_tensor, model, iterations, blocks_deep=None,\
	device=torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" ) ):

	input_tensor.to( device )

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
				]
				original_tile = tile

				if( ( tile.shape[ 2 ] != 299 ) | ( tile.shape[ 3 ] != 299 ) ):
					print(tile.shape)
					print(x_tile)
					print(y_tile)
					print((x_tile+1)*model.image_size+jitter[0])
					print((y_tile+1)*model.image_size+jitter[1])
					utils.show_tensor( tile[0] )

				output = model.dream( tile.to( device ) )
				print( torch.max( output ) )
				print( torch.min( output ) )
				utils.show_tensor( output[ 0 ] )
				utils.show_tensor( tile[ 0 ] )

				input_tensor[\
					:,\
					:,\
					x_tile*model.image_size+jitter[0]:(x_tile+1)*model.image_size+jitter[0],\
					y_tile*model.image_size+jitter[1]:(y_tile+1)*model.image_size+jitter[1]\
				] += output.detach().cpu()


	utils.show_tensor( input_tensor.mean( 0 ) )

def main( args ):

	if( args.image == None ):
		raise Exception( '--image argument required' )
	if( args.iterations == None ):
		args.iterations = 70

	device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
	model = Classifier()

	try:
		model.load_state_dict( torch.load( './classifier.pt' ) )
		print( 'Loading existing model' )
	except:
		raise Exception( 'Model not found' )
	model = model.to( device )

	img = cv2.imread( args.image )
	img_tensor = utils.image_to_tensor( img )
	classize_image( img_tensor, model, args.iterations, blocks_deep=6 )


if __name__ == '__main__':

	parser = argparse.ArgumentParser( description='Transform an input image to look more like an example image' )
	parser.add_argument( '--image', type=str, help='a url pointing to a png image bigger than 299 x 299 pixels' )
	parser.add_argument( '--iterations', type=str, help='number of times to run the image through processing' )
	args = parser.parse_args()

	main( args )

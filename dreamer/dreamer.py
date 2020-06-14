#!/usr/bin/python3

import argparse
import torch
import deep_dream.libutensor as utils
import deep_dream.classifier.model as model
import math
import random
import copy
import cv2
import numpy as np

class Dreamer:
	def __init__( self ):
		self.device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
		self.model = model.Classifier().to( self.device )
		self.tile_size = 300
		self.resolution_ratio = 0.8
		self.iterations = 200
		self.grad_affect = 0.01

	def get_octaves( self, image_shape ):
		remaining_dimensions = [ image_shape[ 0 ], image_shape[ 1 ] ]
		octaves = []
		while( True ):
			if( ( remaining_dimensions[ 0 ] > self.tile_size ) & ( remaining_dimensions[ 1 ] > self.tile_size ) ):
				octaves.append( copy.copy( remaining_dimensions ) )
				remaining_dimensions[ 0 ] = math.floor( remaining_dimensions[ 0 ] * self.resolution_ratio )
				remaining_dimensions[ 1 ] = math.floor( remaining_dimensions[ 1 ] * self.resolution_ratio )
			else:
				break
		return( octaves )

	def get_image_gradient( self, input_img ):
		octaves = self.get_octaves( input_img.shape )
		gradient = np.zeros( input_img.shape )
		for octave in octaves:
			img = cv2.resize( input_img, tuple( [octave[ 1 ], octave[ 0 ]] ), interpolation = cv2.INTER_AREA)
			img_tensor = utils.image_to_tensor( input_img )
			img_tensor = torch.nn.functional.pad(\
				img_tensor,\
				( self.tile_size-1, self.tile_size-1, self.tile_size-1, self.tile_size-1 ),\
				mode='constant',\
				value=0\
			)

			grad_tensor = torch.zeros( img_tensor.shape )

			jitter = ( math.floor( self.tile_size * random.random() ),\
					math.floor( self.tile_size * random.random() ) )
			num_tiles = ( ( math.ceil( ( img_tensor.shape[ 2 ] - jitter[ 0 ] ) / self.tile_size )-1 ),\
					( ( math.ceil( ( img_tensor.shape[ 3 ] - jitter[ 1 ] ) / self.tile_size )-1 ) ) )

			for x_tile in range( 0, num_tiles[ 0 ] ):
				for y_tile in range( 0, num_tiles[ 1 ] ):
					tile = img_tensor[\
						:,\
						:,\
						x_tile*self.tile_size+jitter[0]:(x_tile+1)*self.tile_size+jitter[0],\
						y_tile*self.tile_size+jitter[1]:(y_tile+1)*self.tile_size+jitter[1]\
					].to( self.device )

					tile_gradient = self.model.get_layer_gradient( tile, block=6 )

					grad_tensor[\
						:,\
						:,\
						x_tile*self.tile_size+jitter[0]:(x_tile+1)*self.tile_size+jitter[0],\
						y_tile*self.tile_size+jitter[1]:(y_tile+1)*self.tile_size+jitter[1]\
					] += tile_gradient.cpu().detach()

			grad_tensor = grad_tensor[ :, :, self.tile_size-1:self.tile_size*(-1)+1, self.tile_size-1:self.tile_size*(-1)+1 ]
			grad_tensor = utils.standardize_tensor( grad_tensor )

			rescaled_gradient = cv2.resize(\
				utils.tensor_to_image( grad_tensor[ 0 ] ),\
				tuple( [octaves[ 0 ][ 1 ], octaves[ 0 ][ 0 ]] ),\
				interpolation = cv2.INTER_AREA\
			)

			gradient += rescaled_gradient
		return( gradient )

	def dream( self, input_img ):

		for iter in range( 0, self.iterations ):
			img_gradient = self.get_image_gradient( input_img )
			img_gradient = np.uint8(img_gradient)
			input_img += img_gradient
			utils.show_image( utils.normalize_image( input_img ) )


def main( args ):
	device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
	model = torchvision.models.inception_v3().to( device )
	print( model )

if __name__ == '__main__':

	parser = argparse.ArgumentParser( description='Transform an input image to look more like an example image' )
	args = parser.parse_args()

	main( args )

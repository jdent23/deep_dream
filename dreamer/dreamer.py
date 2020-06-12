#!/usr/bin/python3

import argparse
import torch
import deep_dream.libutensor as utils
import torchvision.models
import math
import random
import copy
import cv2
import numpy as np

class Dreamer:
	def __init__( self ):
		self.device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
		self.model = torchvision.models.inception_v3().to( self.device )
		self.tile_size = 500
		self.resolution_ratio = 0.8
		self.iterations = 20
		self.grad_affect = 0.005

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

	def get_gradient( self, tile ):


		input_tensor = tile.clone().detach().requires_grad_( True )
		x = input_tensor
		y = self.model(x) #.Mixed_7c.branch3x3_1.conv( x )
		loss = utils.dream_loss( y.logits.to( self.device ) ).cpu()
		loss.backward()
		gradient = input_tensor.grad
		gradient = utils.standardize_tensor( gradient )

		return( gradient )

	def dream( self, input_img ):

		input_img = utils.normalize_image( input_img )
		octaves = self.get_octaves( input_img.shape )
		output_img = input_img
		for iter in range( 0, self.iterations ):

			for octave in octaves:
				input_img = cv2.resize( output_img, tuple( [octave[ 1 ], octave[ 0 ]] ), interpolation = cv2.INTER_AREA)
				img_tensor = utils.image_to_tensor( input_img ).to( self.device )
				img_tensor = torch.nn.functional.pad(\
					img_tensor,\
					( self.tile_size-1, self.tile_size-1, self.tile_size-1, self.tile_size-1 ),\
					mode='constant',\
					value=0\
				)


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
						]

						gradient = self.get_gradient( tile.clone() )

						img_tensor[\
							:,\
							:,\
							x_tile*self.tile_size+jitter[0]:(x_tile+1)*self.tile_size+jitter[0],\
							y_tile*self.tile_size+jitter[1]:(y_tile+1)*self.tile_size+jitter[1]\
						] = gradient.detach()

						#print('tile_max: ' + str( torch.max( tile ).item() ) + '\t' + 'tile_min: ' + str( torch.min( tile ).item() ) )
						#print('grad_max: ' + str( torch.max( gradient ).item() ) + '\t' + 'grad_min: ' + str( torch.min( gradient ).item() ) )

				img_tensor = img_tensor[ :, :, self.tile_size-1:self.tile_size*(-1)+1, self.tile_size-1:self.tile_size*(-1)+1 ]
				rescaled_output = cv2.resize(\
					utils.tensor_to_image( img_tensor[ 0 ] ),\
					tuple( [octaves[ 0 ][ 1 ], octaves[ 0 ][ 0 ]] ),\
					interpolation = cv2.INTER_AREA\
				)
				output_img = output_img * ( 1 - self.grad_affect ) + rescaled_output * self.grad_affect
				output_img = utils.normalize_image( output_img )
			utils.show_image( output_img )


def main( args ):
	device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
	model = torchvision.models.inception_v3().to( device )
	print( model )

if __name__ == '__main__':

	parser = argparse.ArgumentParser( description='Transform an input image to look more like an example image' )
	args = parser.parse_args()

	main( args )

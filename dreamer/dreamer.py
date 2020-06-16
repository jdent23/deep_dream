#!/usr/bin/python3

import argparse
import torch
import deep_dream.libutensor as utils
import deep_dream.classifier.model as model
import torchvision.models
import math
import random
import copy
import cv2
import numpy as np

class Dreamer:
	def __init__( self ):
		self.device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
		self.model = torchvision.models.inception_v3( pretrained=True ).to( self.device ) #model.Classifier().to( self.device )
		self.model.eval()
		self.tile_size = 200
		self.tile_crop_margin = 17
		self.resolution_ratio = 0.8
		self.iterations = 30
		self.grad_effect = 0.0002

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

	def get_tile_gradient( self, tensor_tile, block=None ):
		tensor_tile = tensor_tile.clone().detach().requires_grad_( True )
		x = tensor_tile

		if( block == None ):
			block = 16
		while( True ):
			if( block == 0):
				break
			x = self.model.Conv2d_1a_3x3( x )
			if( block == 1):
				break
			x = self.model.Conv2d_2a_3x3( x )
			if( block == 2):
				break
			x = self.model.Conv2d_2b_3x3( x )
			if( block == 3):
				break
			x = self.model.Conv2d_3b_1x1( x )
			if( block == 4):
				break
			x = self.model.Conv2d_4a_3x3( x )
			if( block == 5):
				break
			x = self.model.Mixed_5b( x )
			if( block == 6):
				break
			x = self.model.Mixed_5c( x )
			if( block == 7):
				break
			x = self.model.Mixed_5d( x )
			if( block == 8):
				break
			x = self.model.Mixed_6a( x )
			if( block == 9):
				break
			x = self.model.Mixed_6b( x )
			if( block == 10):
				break
			x = self.model.Mixed_6c( x )
			if( block == 11):
				break
			x = self.model.Mixed_6d( x )
			if( block == 12):
				break
			x = self.model.Mixed_6e( x )
			if( block == 13):
				break
			x = self.model.Mixed_7a( x )
			if( block == 14):
				break
			x = self.model.Mixed_7b( x )
			if( block == 15):
				break
			x = self.model.Mixed_7c( x )
			break

		loss = utils.dream_loss( x )
		loss.backward()
		gradient = tensor_tile.grad

		return gradient

	def get_image_gradient( self, input_img, block=None ):
		octaves = self.get_octaves( input_img.shape )
		gradient = np.zeros( input_img.shape )
		for octave in octaves:
			img = cv2.resize( input_img, tuple( [octave[ 1 ], octave[ 0 ]] ), interpolation = cv2.INTER_AREA)
			img_tensor = utils.image_to_tensor( img )
			img_tensor = torch.nn.functional.pad(\
				img_tensor,\
				( self.tile_size-1, self.tile_size-1, self.tile_size-1, self.tile_size-1 ),\
				mode='reflect'\
			)

			grad_tensor = torch.zeros( img_tensor.shape )

			jitter = ( math.floor( ( self.tile_size - self.tile_crop_margin ) * random.random() ),\
					math.floor( ( self.tile_size - self.tile_crop_margin ) * random.random() ) )
			num_tiles = ( math.floor( ( img_tensor.shape[ 2 ] - jitter[ 0 ] - 2 * self.tile_crop_margin ) / ( self.tile_size - 2 * self.tile_crop_margin ) ),\
					math.floor( ( img_tensor.shape[ 3 ] - jitter[ 1 ] - 2 * self.tile_crop_margin ) / ( self.tile_size - 2 * self.tile_crop_margin ) ) )

			for x_tile in range( 0, num_tiles[ 0 ] ):
				for y_tile in range( 0, num_tiles[ 1 ] ):
					x_start = jitter[0]+x_tile*(self.tile_size-2*self.tile_crop_margin)
					y_start = jitter[1]+y_tile*(self.tile_size-2*self.tile_crop_margin)
					tile = img_tensor[\
						:,\
						:,\
						x_start:x_start+self.tile_size,\
						y_start:y_start+self.tile_size\
					].to( self.device )

					tile_gradient = self.get_tile_gradient( tile, block=block ) # self.model.get_layer_gradient( tile, block=6 )

					cut_gradient = tile_gradient[
						:,\
						:,\
						self.tile_crop_margin:self.tile_crop_margin*(-1),\
						self.tile_crop_margin:self.tile_crop_margin*(-1),\
					].cpu().detach()

					grad_tensor[\
						:,\
						:,\
						x_start+self.tile_crop_margin:x_start+self.tile_size-self.tile_crop_margin,\
						y_start+self.tile_crop_margin:y_start+self.tile_size-self.tile_crop_margin\
					] += cut_gradient

			grad_tensor = grad_tensor[ :, :, self.tile_size-1:self.tile_size*(-1)+1, self.tile_size-1:self.tile_size*(-1)+1 ]
			grad_tensor = utils.standardize_tensor( grad_tensor )

			rescaled_gradient = cv2.resize(\
				utils.tensor_to_image( grad_tensor[ 0 ] ),\
				tuple( [octaves[ 0 ][ 1 ], octaves[ 0 ][ 0 ]] ),\
				interpolation = cv2.INTER_AREA\
			)

			gradient += rescaled_gradient
		return( gradient )

	def dream( self, input_img, block=None ):

		input_img = utils.normalize_image( np.float32( input_img ) )
		for iter in range( 0, self.iterations ):
			img_gradient = self.get_image_gradient( input_img.copy(), block )
			img_gradient = np.float32(img_gradient)
			input_img += img_gradient * self.grad_effect
			input_img = utils.normalize_image( input_img )
		return( input_img )


def main( args ):
	device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
	model = torchvision.models.inception_v3( pretrained=True ).to( device )
	print(model)

if __name__ == '__main__':

	parser = argparse.ArgumentParser( description='Transform an input image to look more like an example image' )
	args = parser.parse_args()

	main( args )

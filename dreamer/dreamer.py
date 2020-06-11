#!/usr/bin/python3

import argparse
import torch
import deep_dream.libutensor as utils
import torchvision.models
import math
import random

class Dreamer:
	def __init__( self ):
		self.device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
		self.model = torchvision.models.inception_v3().to( self.device )
		self.tile_size = 500

	def get_gradient( self, tile ):

		for color_channel in range( 0, 3 ): # self.model.Mixed_7c.branch3x3_1.conv.state_dict()[ 'weight' ].shape[ 1 ]
			tensor_channel = tile[ 0, color_channel, :, : ]
			input_tensor = tensor_channel[None, None, :, :].repeat( 1,\
				3,\
				1,\
				1\
			).clone().detach().requires_grad_( True )
			x = input_tensor
			x = self.model(x) #.Mixed_7c.branch3x3_1.conv( x )
			print(x)
			loss = utils.dream_loss( x.logits.to( self.device ) ).cpu()
			loss.backward()
			gradient = input_tensor.grad
			gradient = utils.standardize_tensor( gradient )
			tile[ 0, color_channel, :, : ] = torch.mean( gradient, ( 0, 1 ) )
		print(torch.max(tile))
		print(torch.min(tile))
		utils.show_tensor(torch.abs(tile[0])*12)

		return( tile )

	def dream( self, input_img, iterations=1 ):

		gradient_affect = 0.2

		img_tensor = utils.image_to_tensor( input_img ).to( self.device )
		img_tensor = torch.nn.functional.pad(\
			img_tensor,\
			( self.tile_size-1, self.tile_size-1, self.tile_size-1, self.tile_size-1 ),\
			mode='constant',\
			value=0\
		)

		for iter in range( 0, iterations ):

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
					original_tile = tile

					if( ( tile.shape[ 2 ] != 299 ) | ( tile.shape[ 3 ] != 299 ) ):
						print(tile.shape)
						print(x_tile)
						print(y_tile)
						print((x_tile+1)*self.tile_size+jitter[0])
						print((y_tile+1)*self.tile_size+jitter[1])
						utils.show_tensor( tile[0] )

					gradient = self.get_gradient( tile )

					img_tensor[\
						:,\
						:,\
						x_tile*self.tile_size+jitter[0]:(x_tile+1)*self.tile_size+jitter[0],\
						y_tile*self.tile_size+jitter[1]:(y_tile+1)*self.tile_size+jitter[1]\
					] += gradient.detach()


		#utils.show_tensor( input_tensor.mean( 0 ) )


def main( args ):
	device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
	model = torchvision.models.inception_v3().to( device )
	print( model )

if __name__ == '__main__':

	parser = argparse.ArgumentParser( description='Transform an input image to look more like an example image' )
	args = parser.parse_args()

	main( args )

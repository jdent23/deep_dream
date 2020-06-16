#!/usr/bin/python3

import torch
from torch.utils import data
import cv2
import argparse
import random
import math
import torchvision.models
import deep_dream.dreamer
import numpy as np
import deep_dream.libutensor as utils

def zoom_image( img, resolution, zoom_ratio ):
	img = cv2.resize(img,None,fx=1.01, fy=1.01, interpolation = cv2.INTER_LINEAR)
	x_start = math.floor( ( img.shape[ 0 ] - resolution[ 0 ] ) / 2 )
	y_start = math.floor( ( img.shape[ 1 ] - resolution[ 1 ] ) / 2 )
	img = img[\
		x_start:x_start+resolution[0],\
		y_start:y_start+resolution[1],\
		:\
	]
	return( img )

def main( args ):

	if( args.image == None ):
		raise Exception( '--image argument required' )

	zoom_ratio = 1.01
	resolution = ( 480, 640 )
	img = cv2.imread( args.image )
	img = zoom_image( img, resolution, zoom_ratio )

	write_path = '/home/jasondent/deep_dream/dreamer/fractal_frames/'
	dreamer = deep_dream.dreamer.Dreamer()

	utils.save_image( img, write_path + '0.png' )

	frame = 0
	for block in range( 15, 0, -1 ):
		for transition in range( 0, 360 ):
			frame += 1
			print( 'Starting frame ' + str( frame ) + ' for block ' + str( block ) + ', transition ' + str( transition ) )
			img = zoom_image( img, resolution, zoom_ratio )
			img = dreamer.dream( img, block=block )
			utils.save_image( img, write_path + str( frame ) + '.png' )


if __name__ == '__main__':

	parser = argparse.ArgumentParser( description='Transform an input image to look more like an example image' )
	parser.add_argument( '--image', type=str, help='a url pointing to a png image bigger than 299 x 299 pixels' )
	args = parser.parse_args()

	main( args )

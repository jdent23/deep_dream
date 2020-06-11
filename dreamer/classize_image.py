#!/usr/bin/python3

import torch
from torch.utils import data
import cv2
import argparse
import random
import math
import torchvision.models
import deep_dream.dreamer

def main( args ):

	if( args.image == None ):
		raise Exception( '--image argument required' )
	if( args.iterations == None ):
		args.iterations = 70

	img = cv2.imread( args.image )
	dreamer = deep_dream.dreamer.Dreamer()
	dreamer.dream( img )


if __name__ == '__main__':

	parser = argparse.ArgumentParser( description='Transform an input image to look more like an example image' )
	parser.add_argument( '--image', type=str, help='a url pointing to a png image bigger than 299 x 299 pixels' )
	parser.add_argument( '--iterations', type=str, help='number of times to run the image through processing' )
	args = parser.parse_args()

	main( args )

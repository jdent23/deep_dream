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

def main( args ):

	if( args.image == None ):
		raise Exception( '--image argument required' )

	img = cv2.imread( args.image )
	dreamer = deep_dream.dreamer.Dreamer()
	dreamer.dream( img, block=3 )


if __name__ == '__main__':

	parser = argparse.ArgumentParser( description='Transform an input image to look more like an example image' )
	parser.add_argument( '--image', type=str, help='a url pointing to a png image bigger than 299 x 299 pixels' )
	args = parser.parse_args()

	main( args )

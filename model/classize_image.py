#!/usr/bin/python3

import torch
from torch.utils import data
from data_generator import Dataset
from model import Classifier
import cv2
import argparse

def classize_image( input_tensor, example_tensor, model, iterations ):
	optimizer = torch.optim.SGD( model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4 )
	f_loss = torch.nn.MSELoss( size_average=None, reduce=None, reduction='mean' )

	x = model( example_tensor, blocks_deep=0 )
	print(x.shape)

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

	params = {'batch_size': 1,
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

import torch
from torch.utils import data
from data_generator import Dataset
import cv2
import argparse
import random
import math
import sys

def image_to_tensor( input_image ):
	return( torch.from_numpy( input_image ).permute( 2, 0, 1 )[ None, :, :, : ].type( torch.FloatTensor ) / 255 )

def tensor_to_image( input_tensor ):
	return( ( input_tensor ).permute( 1, 2, 0 ).cpu().detach().numpy() )

def show_tensor( input_tensor ):
	cv2.imshow('image', tensor_to_image( input_tensor ) )
	cv2.waitKey(0) & 0xFF
	cv2.destroyAllWindows()

def normalize_tensor( tensor ):
	print( torch.max( tensor ) )
	print( torch.min( tensor ) )
	return( ( tensor - torch.min( tensor ) ) / ( torch.max( tensor ) - torch.min( tensor ) ) )
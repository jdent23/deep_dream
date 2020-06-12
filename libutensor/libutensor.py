import torch
import cv2
import numpy as np

def image_to_tensor( input_image ):
	return( torch.from_numpy( input_image ).permute( 2, 0, 1 )[ None, :, :, : ].type( torch.FloatTensor ) / 255 )

def tensor_to_image( input_tensor ):
	return( ( input_tensor ).permute( 1, 2, 0 ).cpu().detach().numpy() )

def show_image( input_image ):
	cv2.imshow('image', input_image )
	cv2.waitKey(0) & 0xFF
	cv2.destroyAllWindows()

def show_tensor( input_tensor ):
	cv2.imshow('image', tensor_to_image( input_tensor ) )
	cv2.waitKey(0) & 0xFF
	cv2.destroyAllWindows()

def normalize_image( img ):
	return( ( img - np.min( img ) ) / ( np.max( img ) - np.min( img ) ) )

def normalize_tensor( tensor ):
	return( ( tensor - torch.min( tensor ) ) / ( torch.max( tensor ) - torch.min( tensor ) ) )

def standardize_tensor( tensor ):
	return( ( tensor - torch.mean( tensor ) ) / torch.std( tensor ) )

def dream_loss( model_output ):
	return( torch.sum( model_output*(-1) ) )
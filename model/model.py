import torch
import tensor_utils as utils

def dream_loss( model_output ):
	return( torch.sum( model_output*(-1) ) )

class Xception( torch.nn.Module ):
	def __init__( self, in_channels, out_channels ):
		super().__init__()

		self.L = torch.nn.ModuleList( [] )
		self.L.append(\
			torch.nn.Conv2d( in_channels=in_channels,\
				out_channels=out_channels,\
				kernel_size=( 1, 1 ),\
				stride=( 1, 1 ),\
				padding=0,\
				dilation=1,\
				groups=1,\
				bias=True,\
				padding_mode='zeros'\
			)\
		)
		self.L.append(\
			torch.nn.Conv2d( in_channels=out_channels,\
				out_channels=out_channels,\
				kernel_size=( 3, 3 ),\
				stride=( 1, 1 ),\
				padding=1,\
				dilation=1,\
				groups=out_channels,\
				bias=True,\
				padding_mode='zeros'\
			)\
		)

	def forward( self, tensor_img ):
		for layer in self.L:
			tensor_img = layer( tensor_img )

		return( tensor_img )

class Classifier( torch.nn.Module ):
	def __init__( self ):
		super().__init__()

		self.image_size = 299
		self.dropout = 0.5
		self.channels = [ 16, 32, 32, 64, 64, 128 ]
		self.preferred_lr = [ 0, 0, 0, 0, 0, 0.1 ]
		self.block_layers = 2
		self.L = torch.nn.ModuleList( [] )
		self.L_residual = torch.nn.ModuleList( [] )
		in_channels = 3
		for channel in self.channels:
			out_channels = channel

			self.L_residual.append(\
				torch.nn.Conv2d( in_channels=in_channels,\
					out_channels=out_channels,\
					kernel_size=( 3, 3 ),\
					stride=( 1, 1 ),\
					padding=( 1, 1 ),\
					dilation=1,\
					groups=1,\
					bias=True,\
					padding_mode='zeros'\
				)\
			)

			for block_layer in range( 0, self.block_layers ):
				self.L.append( Xception( in_channels, out_channels ) )
				in_channels = out_channels
			in_channels = out_channels

		self.relu = torch.nn.ReLU()
		self.max_pool = torch.nn.MaxPool2d(\
			kernel_size=( 3, 3),\
			stride=( 2, 2 ),\
			padding=1,\
			dilation=1,\
			return_indices=False,\
			ceil_mode=False\
		)


		self.classifier = torch.nn.ModuleList( [] )
		self.classifier.append( torch.nn.Linear( 3200, 1600, bias=True) )
		self.classifier.append( torch.nn.ReLU() )
		self.classifier.append( torch.nn.Dropout( p=self.dropout ) )
		self.classifier.append( torch.nn.Linear( 1600, 1, bias=True) )

		self.sigmoid = torch.nn.Sigmoid()

	def forward( self, tensor_img, train=False,\
		device=torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )):

		x = tensor_img
		residual = x

		for layer in range( 0, len( self.L ) ):
			if( ( layer % self.block_layers == 0 ) & ( layer != 0 ) ):
				residual = self.L_residual[ int( ( layer/self.block_layers ) - 1 ) ]( residual )
				residual = self.relu( residual )

				x += residual
				x = self.max_pool( x )
			if( layer % self.block_layers == 0 ):
				residual = x

			x = self.L[ layer ]( x )
			x = self.relu( x )

			if( layer / self.block_layers == blocks_deep ):
				return( x )

		residual = self.L_residual[ int( ( ( layer+1 )/self.block_layers ) - 1 ) ]( residual )
		residual = self.relu( residual )
		x += residual
		x = self.max_pool( x )

		x = torch.flatten(x, start_dim=1)
		for layer in self.classifier:
			x = layer( x )

		if( train == False ):
			x = self.sigmoid( x ).detach()

		return x

	def dream( self, tensor_img, block=6 ):

		if( tensor_img.shape[ 0 ] != 1 ):
			raise Exception( "Expected a tensor of only one image with dimensions[ 1, rgb, height, width ]" )
		if( tensor_img.shape[ 1 ] != 3 ):
			raise Exception( 'Expected tensor_img to have rgb channels only' )

		for color_channel in range( 0, 3 ):
			tensor_channel = tensor_img[ 0, color_channel, :, : ]
			input_tensor = tensor_channel[None, None, :, :].repeat( 1,\
				self.L[ ( block - 1 )*self.block_layers ].state_dict()[ 'L.0.weight' ].shape[ 1 ],\
				1,\
				1\
			).detach().requires_grad_( True )
			x = input_tensor

			for layer in range( ( block - 1 )*self.block_layers, block*self.block_layers-1 ):
				x = self.L[ layer ]( x )

			loss = dream_loss( x )
			loss.backward()
			gradient = input_tensor.grad
			gradient = utils.normalize_tensor( gradient )
			tensor_img[ 0, color_channel, :, : ] = torch.mean( gradient, ( 0, 1 ) )

		return( tensor_img )


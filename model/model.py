import torch

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

		self.dropout = 0.5
		self.channels = [ 16, 32, 32, 64, 64, 128 ]
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

	def forward( self, tensor_img, train=False, blocks_deep=None ):
		if( blocks_deep == None ):
			blocks_deep = -1

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


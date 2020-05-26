#!/usr/bin/python3
import torch

class Xception( torch.nn.Module ):
    def __init__( self, in_channels, out_channels ):
        super().__init__()

        L = torch.nn.ModuleList( [] )
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
    		tensor_img = tensor_img( layer )

    	return( tensor_img )

class Classifier( torch.nn.Module ):
    def __init__( self ):
        super().__init__()

        self.induction_1 = torch.nn.ModuleList( [] )
        self.induction_1.append(\
        	torch.nn.Conv2d( in_channels=3,\
	        	out_channels=32,\
	        	kernel_size=( 3, 3 ),\
	        	stride=( 2, 2 ),\
	        	padding=0,\
	        	dilation=1,\
	        	groups=1,\
	        	bias=True,\
	        	padding_mode='zeros'\
        	)\
        )
        self.induction_1.append( torch.nn.ReLU() )
        self.induction_1.append(\
        	torch.nn.Conv2d( in_channels=32,\
	        	out_channels=64,\
	        	kernel_size=( 3, 3 ),\
	        	stride=( 1, 1 ),\
	        	padding=0,\
	        	dilation=1,\
	        	groups=1,\
	        	bias=True,\
	        	padding_mode='zeros'\
        	)\
        )
        self.induction_1.append( torch.nn.ReLU() )

        self.induction_2 = torch.nn.ModuleList( [] )
        self.induction_2.append( Xception( 64, 128 ) )
        self.induction_2.append( torch.nn.ReLU() )
        self.induction_2.append( Xception( 128, 128 ) )
        self.induction_2.append(\
        	torch.nn.MaxPool2d(\
	        	kernel_size=( 3, 3),\
	        	stride=( 2, 2 ),\
	        	padding=1,\
	        	dilation=1,\
	        	return_indices=False,\
	        	ceil_mode=False\
        	)\
        )

        self.induction_2_pass = torch.nn.ModuleList( [] )
        self.induction_2_pass.append(\
        	torch.nn.Conv2d( in_channels=3,\
	        	out_channels=128,\
	        	kernel_size=( 1, 1 ),\
	        	stride=( 2, 2 ),\
	        	padding=0,\
	        	dilation=1,\
	        	groups=1,\
	        	bias=True,\
	        	padding_mode='zeros'\
        	)\
        )

        self.induction_3 = torch.nn.ModuleList( [] )
        self.induction_3.append( torch.nn.ReLU() )
        self.induction_3.append( Xception( 128, 256 ) )
        self.induction_3.append( torch.nn.ReLU() )
        self.induction_3.append( Xception( 256, 256 ) )
        self.induction_3.append(\
        	torch.nn.MaxPool2d(\
	        	kernel_size=( 3, 3),\
	        	stride=( 2, 2 ),\
	        	padding=1,\
	        	dilation=1,\
	        	return_indices=False,\
	        	ceil_mode=False\
        	)\
        )

        self.induction_3_pass = torch.nn.ModuleList( [] )
        self.induction_3_pass.append(\
        	torch.nn.Conv2d( in_channels=128,\
	        	out_channels=256,\
	        	kernel_size=( 1, 1 ),\
	        	stride=( 2, 2 ),\
	        	padding=0,\
	        	dilation=1,\
	        	groups=1,\
	        	bias=True,\
	        	padding_mode='zeros'\
        	)\
        )

        self.induction_4 = torch.nn.ModuleList( [] )
        self.induction_4.append( torch.nn.ReLU() )
        self.induction_4.append( Xception( 256, 728 ) )
        self.induction_4.append( torch.nn.ReLU() )
        self.induction_4.append( Xception( 728, 728 ) )
        self.induction_4.append(\
        	torch.nn.MaxPool2d(\
	        	kernel_size=( 3, 3),\
	        	stride=( 2, 2 ),\
	        	padding=1,\
	        	dilation=1,\
	        	return_indices=False,\
	        	ceil_mode=False\
        	)\
        )

        self.induction_4_pass = torch.nn.ModuleList( [] )
        self.induction_4_pass.append(\
        	torch.nn.Conv2d( in_channels=256,\
	        	out_channels=728,\
	        	kernel_size=( 1, 1 ),\
	        	stride=( 2, 2 ),\
	        	padding=0,\
	        	dilation=1,\
	        	groups=1,\
	        	bias=True,\
	        	padding_mode='zeros'\
        	)\
        )


        # Middle
        self.middle_L = torch.nn.ModuleList( [] )

        for i in range( 0, 8 ):
        	for j in range( 0, 3 ):
		        self.induction_4.append( torch.nn.ReLU() )
		        self.induction_4.append( Xception( 728, 728 ) )

		# finish

		self.finish_1 = torch.nn.ModuleList( [] )
        self.finish_1.append( torch.nn.ReLU() )
        self.finish_1.append( Xception( 256, 728 ) )
        self.finish_1.append( torch.nn.ReLU() )
        self.finish_1.append( Xception( 728, 728 ) )
        self.finish_1.append(\
        	torch.nn.MaxPool2d(\
	        	kernel_size=( 3, 3),\
	        	stride=( 2, 2 ),\
	        	padding=1,\
	        	dilation=1,\
	        	return_indices=False,\
	        	ceil_mode=False\
        	)\
        )

        self.finish_1_pass = torch.nn.ModuleList( [] )
        self.finish_1_pass.append(\
        	torch.nn.Conv2d( in_channels=256,\
	        	out_channels=728,\
	        	kernel_size=( 1, 1 ),\
	        	stride=( 2, 2 ),\
	        	padding=0,\
	        	dilation=1,\
	        	groups=1,\
	        	bias=True,\
	        	padding_mode='zeros'\
        	)\
        )

        self.finish_2 = torch.nn.ModuleList( [] )

        self.finish_2.append( Xception( 728, 1536 ) )
        self.finish_2.append( torch.nn.ReLU() )
        self.finish_2.append( Xception( 1536, 2048 ) )
        self.finish_2.append( torch.nn.ReLU() )
        self.finish_2.append(\
	        torch.nn.AvgPool2d(kernel_size,\
	        	stride=None,\
	        	padding=0,\
	        	ceil_mode=False,\
	        	count_include_pad=True,\
	        	divisor_override=None\
	        )
	    )

        self.classifier = torch.nn.ModuleList( [] )
        self.classifier.append( torch.nn.Linear( 4000, 300, bias=True) )
        self.classifier.append( torch.nn.Linear( 300, 1, bias=True) )

    def forward(self, tensor_img):
    	pass

    def predict(self, tensor_img):
    	pass

def main():
        print("hello")

if __name__ == '__main__': 
        main()

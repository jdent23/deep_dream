import torch
from torch.utils import data
import os
import cv2
import random
import math

class Dataset( data.Dataset ):
    def __init__( self, data_dir, class_names_list, image_size ):

        self.image_size = image_size
        self.data_dir = data_dir + '/'
        self.class_names_list = class_names_list


        self.list_IDs = []
        self.labels = {}
        index = -1
        for class_name in self.class_names_list:
            index += 1

            class_files = os.listdir( self.data_dir + class_name )

            for file in class_files:

                path = self.data_dir + class_name + '/' + file
                #img = cv2.imread( path )
                #if( ( img.shape[ 0 ] < self.image_size ) | ( img.shape[ 1 ] < self.image_size ) ):
                #    os.remove( path )
                #    print( 'Remove file: ' + path + '; image too small' )
                #    continue

                label = []
                for i in range( 0, len( self.class_names_list ) ):
                    if( i == index ):
                        label.append( 1 )
                    else:

                        label.append( 0 )

                self.list_IDs.append( path )
                self.labels[ path ] = label

    def __len__( self ):
        return( len( self.list_IDs ) )

    def __getitem__( self, index ):
        ID = self.list_IDs[ index ]

        img = cv2.imread( ID )
        img_tensor = torch.from_numpy( img )
        y_offset =  math.floor( random.random() * ( img_tensor.shape[ 0 ] - self.image_size ) )
        x_offset =  math.floor( random.random() * ( img_tensor.shape[ 1 ] - self.image_size ) )
        img_tensor_cut = img_tensor[\
            y_offset:self.image_size+y_offset,\
            x_offset:self.image_size+x_offset,\
            :\
        ]

        y = torch.Tensor( self.labels[ ID ] )

        return( img_tensor_cut.permute( 2, 0, 1 ).type( torch.FloatTensor ) / 255, y )
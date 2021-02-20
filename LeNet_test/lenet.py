from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K 

class LeNet:
    @staticmethod
    def build(width,height,depth,classes):
        model = Sequential(0)
        inputShape = (height,width,depth)


        '''
            舉個example 
            原始圖片為28x28的RGB
            如果是channels_first 就會變成 3x28x28的擺法
        '''
        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)

        # start stacking our Net

        # fist CONV 
        model.add(Conv2D(20,(5,5),padding='same',input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2),strides=(2,2)))

        # second CONV 
        model.add(Conv2D(50,(5,5),padding='same',input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPool2D(pool_size=(2, 2),strides=(2,2)))

        # Fully-connected NetWork 
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # 0~9 conditions so 10 
        model.add(Dense(10))
        model.add(Activation('softmax'))

        return model 

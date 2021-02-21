from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K 


class MiniVGGNet:
    @staticmethod
    def build(width,height,depth,classes):

        inputShape = (height,width,depth)
        chanDim = -1 # for BatchNormalization

        if K.image_data_format == 'channel_first':
            inputShape = (depth,height,width)
            chanDim = 1 

        model = Sequential()

        # first layer
        model.add(Conv2D(32,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        # second layer
        model.add(Conv2D(64,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # flatten to 1-d array 
        model.add(Flatten())

        # Fully-connected network
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model





        return model 
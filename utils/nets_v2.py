"""
	Author: Ahad Suleymanli

    A two-imput day-night classification model

"""
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import concatenate
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras import backend as K

class LeNet_v2:
    @staticmethod
    def build(width, height, depth, feature_shape, classes):
        '''
            first 3 parameters are for the input image
            feature_shape is the shape of something 
                    like a intensity histogram of the picture
            classes is the number of output classes
        '''
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        else:
            inputShape = (height, width, depth)
        x_in = Input(shape=inputShape, name="x_in")
        y_in = Input(shape=feature_shape, name="y_in")

        # first set of CONV => RELU => POOL layers
        x = Conv2D(20, (5, 5), padding="same", activation="relu")(x_in)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        # second set of CONV => RELU => POOL layers
        x = Conv2D(50, (5, 5), padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        # first (and only) set of FC => RELU layers
        # also concatenates the injected feature vector with x
        x = Flatten()(x)
        y = Dense(200, activation="relu")(y_in)
        z = concatenate([x, y])  
        z = Dense(500, activation="relu")(z)



        # softmax classifier
        output = Dense(classes, activation="softmax", name="day_night")(z)

        # create the Model with the specified input and outputs
        model = Model(inputs=[x_in, y_in], outputs=output, name="day_night_model")

        return model
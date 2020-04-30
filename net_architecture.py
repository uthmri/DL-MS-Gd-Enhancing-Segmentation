import sys
from functools import partial
from keras.engine import Model
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, Deconvolution3D, BatchNormalization


# function which creates block including two 3D convolutional layers
def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):

    new_layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    layer = Conv3D(n_filters, kernel, padding=padding, strides=(1, 1, 1))(new_layer)
    if batch_normalization:
        layer = BatchNormalization()(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization.")
        layer = InstanceNormalization()(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

# converts original function to a simpler version with specified inputs
create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


# function to create model upon specified parameters
def i_model(input_shape=(128, 128, 16, 5), n_base_filters=64, depth=4, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=6, activation_name="softmax"):

    # Defines input layer to model
    inputs = Input(input_shape)


    current_layer, level_output_layers, level_filters = inputs, list(), list()

    # for loop to handle creation of context modules
    for level_number in range(depth):

        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        # first convolution stage process images at original resolution
        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)

        # subsequent stages would reduce original resolution by striding = 2 on all axes
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        # creates residual connection between stages
        summation_layer = Add()([in_conv, context_output_layer])

        level_output_layers.append(summation_layer)

        current_layer = summation_layer

    # Loop to handle definition of expanding path and localization module
    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):

        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])

        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=-1)

        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output

        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None


    # Loop to handle definition of upsampling layers
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2 ,2 ,2))(output_layer)


    activation_block = Activation(activation_name)(output_layer)

    # defines whole model structure based on defined layers
    model = Model(inputs=inputs, outputs=activation_block)

    # prints summary of model layers
    model.summary()


    return model



# creates localization module which includes some convolutional layers
def create_localization_module(input_layer, n_filters):

    # convolutional layers to aggregate segmentations between feature levels
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))

    return convolution2


# creates upsampling module which includes convolutional layers and deconvolutional layers
def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):

    # upsamples network internal expanding path resolution to match resolution of contracting path
    up_sample = Deconvolution3D(filters=n_filters, kernel_size=(2, 2, 2), strides=(2, 2, 2))(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)

    return convolution


# creates context module encompasing convolutional layers and feature dropout
def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_last"):

    # creates convolutional block to process image features
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    # dropouts some weight to reduce overfitting of features
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    # Additional convolutional block to process more features and manage feature maps
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2










def main():
    i_model()

if __name__ == '__main__':
    main()

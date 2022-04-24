all_dict = {}

dense_dict = {
    "units": [100, 200, 300, 400, 500]
}

conv2d_dict = {
#     "filters": [4, 8, 12, 16, 20],
#     "strides": [(1,1), (2,2), (3,3), (4,4), (5,5)],
    "padding": ['valid', 'same'], # ok
    "activation": ['linear', 'relu', 'tanh', 'softmax'] # ok
}

batchnorm_dict = {
    "momentum": [0, 0.2, 0.4, 0.6, 0.8],
    "epsilon": [0, 0.2, 0.4, 0.6, 0.8]
}

maxpooling2d_dict = {
#     "strides": [(1,1), (2,2), (3,3),(4,4), (5,5)],
    "padding": ['valid', 'same']
}

##
dropout_dict = {
    "rate": [0, 0.2, 0.4, 0.6, 0.8],
    "seed": [1, 2, 3, 4, 5]
}

activation_dict = {
    "activation": ['linear', 'relu', 'tanh', 'softmax']
}

zeropadd2d_dict = {
    "padding": [(1,1), (2,2), (3,3), (4,4), (5,5)],
}

avergpool1d_dict = {
    "strides": [1, 2, 3, 4, 5],
    "padding": ['valid', 'same']    
}

avergpool2d_dict = {
    "strides": [(1,1), (2,2), (3,3), (4,4), (5,5)],
    "padding": ['valid', 'same']
}

avergpool3d_dict = {
    "strides": [(1,1,1), (2,2,2), (3,3,3), (4, 4, 4), (5,5,5)],
    "padding": ['valid', 'same']
}

conv1d_dict = {
    "filters": [4, 8, 12, 16, 20],
    "strides": [1, 2, 3, 4, 5],
    "padding": ['valid', 'same'],
    "activation": ['linear', 'relu', 'tanh', 'softmax']
}

conv3d_dict = {
    "filters": [4, 8, 12, 16, 20],
    "strides": [(1,1,1), (2,2,2), (3,3,3), (4,4,4), (5,5,5)],
    "padding": ['valid', 'same'],
    "activation": ['linear', 'relu', 'tanh', 'softmax']
}

localconnect1d_dict = {
    "filters": [4, 8, 12, 16, 20],
    "strides": [1, 2, 3, 4, 5],
    "activation": ['linear', 'relu', 'tanh', 'softmax']
}

localconnect2d_dict = {
    "filters": [4, 8, 12, 16, 20],
    "strides": [(1,1), (2,2), (3,3), (4,4), (5,5)],
    "activation": ['linear', 'relu', 'tanh', 'softmax']
}

relu_dict = {
    "max_value": [0, 0.2, 0.4, 0.6, 0.8]
}

repeatvector_dict = {
    "n": [0, 200, 400, 600, 800]
}

separableconv1d_dict = {
    "filters": [4, 8, 12, 16, 20],
    "strides": [1, 2, 3, 4, 5],
    "padding": ['valid', 'same', 'causal'],
    "activation": ['linear', 'relu', 'tanh', 'softmax']
}

separableconv2d_dict = {
    "filters": [4, 8, 12, 16, 20],
    "strides": [(1,1), (2,2), (3,3), (4,4), (5,5)],
    "padding": ['valid', 'same', 'causal'],
    "activation": ['linear', 'relu', 'tanh', 'softmax']
}

simplernn_dict = {
    "units": [0, 200, 400, 600, 800],
    "activation": ['linear', 'relu', 'tanh', 'softmax'],
    "use_bias": [True, False],
    "dropout": [0, 0.2, 0.4, 0.6, 0.8],
    "recurrent_dropout": [0, 0.2, 0.4, 0.6, 0.8],
    "return_sequences": [True, False]
}

spacialdropout1d_dict = {
    "rate": [0, 0.2, 0.4, 0.6, 0.8]
}

spacialdropout2d_dict = {
    "rate": [0, 0.2, 0.4, 0.6, 0.8]
}

spacialdropout3d_dict = {
    "rate": [0, 0.2, 0.4, 0.6, 0.8]
}

thresholdrelu_dict = {
    "theta": [0, 0.2, 0.4, 0.6, 0.8]
}

conv2dtranspose_dict = {
    "filters": [4, 8, 12, 16, 20],
    "strides": [(1,1), (2,2), (3,3), (4,4), (5,5)],
    "padding": ['valid', 'same'],
    "activation": ['linear', 'relu', 'tanh', 'softmax']
}

conv3dtranspose_dict = {
    "filters": [4, 8, 12, 16, 20],
    "strides": [(1,1,1), (2,2,2), (3,3,3), (4,4,4), (5,5,5)],
    "padding": ['valid', 'same'],
    "activation": ['linear', 'relu', 'tanh', 'softmax']
}

depthwiseconv2d_dict = {
    "strides": [(1,1), (2,2), (3,3), (4,4), (5,5)],
    "padding": ['valid', 'same'],
    "activation": ['linear', 'relu', 'tanh', 'softmax']
}

gru_dict = {
    "units": [200, 400, 600, 800, 1000],
#     "activation": ['linear', 'relu', 'tanh', 'softmax'],
    "use_bias": [True, False]
}

grucell_dict = {
    "units": [200, 400, 600, 800, 1000],
    "activation": ['linear', 'relu', 'tanh', 'softmax'],
    "use_bias": [True, False]
}

gaussiandropout_dict = {
    "rate": [0, 0.2, 0.4, 0.6, 0.8]
}

gaussiannoice_dict = {
    "stddev": [0, 0.2, 0.4, 0.6, 0.8]
}

lstm_dict = {
    "units": [0, 200, 400, 600, 800],
    "activation": ['linear', 'relu', 'tanh', 'softmax'],
    "use_bias": [True, False],
    "dropout": [0, 0.2, 0.4, 0.6, 0.8],
    "recurrent_dropout": [0, 0.2, 0.4, 0.6, 0.8],
    "return_sequences": [True, False],
    "go_backwards": [True, False],
    "time_major": [True, False]
}


all_dict['Dense'] = dense_dict
all_dict['Conv2D'] = conv2d_dict
all_dict['BatchNormalization'] = batchnorm_dict
all_dict['MaxPooling2D'] = maxpooling2d_dict
all_dict['Dropout'] = dropout_dict
all_dict['Activation'] = activation_dict
all_dict['ZeroPadding2D'] = zeropadd2d_dict
all_dict['AveragePooling1D'] = avergpool1d_dict
all_dict['AveragePooling2D'] = avergpool2d_dict
all_dict['AveragePooling3D'] = avergpool3d_dict 
all_dict['LocallyConnected1D'] = localconnect1d_dict
all_dict['LocallyConnected2D'] = localconnect2d_dict
all_dict['ReLU'] = relu_dict
all_dict['RepeatVector'] = repeatvector_dict
all_dict['SeparableConv1D'] = separableconv1d_dict
all_dict['SeparableConv2D'] = separableconv2d_dict
all_dict['SimpleRNN'] = simplernn_dict
all_dict['SpatialDropout1D'] = spacialdropout1d_dict
all_dict['SpatialDropout2D'] = spacialdropout2d_dict
all_dict['SpatialDropout3D'] = spacialdropout3d_dict 
all_dict['ThresholdedReLU'] = thresholdrelu_dict
all_dict['Conv2DTranspose'] = conv2dtranspose_dict
all_dict['Conv3DTranspose'] = conv3dtranspose_dict
all_dict['DepthwiseConv2D'] = depthwiseconv2d_dict
all_dict['GRU'] = gru_dict
all_dict['GRUCell'] = grucell_dict
all_dict['GaussianDropout'] = gaussiandropout_dict
all_dict['GaussianNoise'] = gaussiannoice_dict
all_dict['LSTM'] = lstm_dict

def get_configs(model):
    list1 = []
    layer_str = "layer_{0}"
    num = 0
    # model.layers[i]
    for i in model.layers:          
        cur_layer_name = i.__class__.__name__        
        cur_dict = all_dict.get(cur_layer_name)
        
        avail_dict = {}
        if cur_dict!=None:
            # for each key in each layer
            for j in cur_dict.keys():
                cur_param = i.get_config()[j]
                values = cur_dict.get(j)
                if cur_param in values:
                    values.remove(cur_param)
                avail_dict[j] = values
        
        list1.append(avail_dict)
        num += 1
    return list1
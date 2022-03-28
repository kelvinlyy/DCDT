import tensorflow as tf


def addDense(name, config = tf.keras.layers.Dense(units=1, input_shape=[1]).get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0] % 5))
    return tf.keras.layers.Dense.from_config(config)

def addConv2D(name, config = tf.keras.layers.Conv2D(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid','exponential', 'linear'][seed[3] % 11]
    
    return tf.keras.layers.Conv2D.from_config(config)
    
def addBatchNorm(name, config = tf.keras.layers.BatchNormalization().get_config(), seed = [0,0]):
    config["name"] = f'layer_{name}'
    config["momentum"] = 1/5*(seed[0] % 5)
    config["epsilon"] = 1/5*(seed[1] % 5)
    
    return tf.keras.layers.BatchNormalization.from_config(config)
    
    
def addMaxPool(name, config = tf.keras.layers.MaxPooling2D(pool_size=(1,1)).get_config(), seed = [0,0]):
    config["name"] = f'layer_{name}'
    config["strides"] = [int(5/5*(seed[0] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[1] % 2]
    
    return tf.keras.layers.MaxPooling2D.from_config(config)
    

def addDropout(name, config = tf.keras.layers.Dropout(rate=0.1).get_config(), seed = [0,0]):
    config["name"] = f'layer_{name}'
    config["rate"] = 1/5*(seed[0] % 5)
    config["seed"] = seed[1] % 5
    
    return tf.keras.layers.Dropout.from_config(config)

def addActivation(name, config = tf.keras.layers.Activation(activation="softmax").get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'][seed[0] % 11]
    
    return tf.keras.layers.Activation.from_config(config)

def addZeroPadd(name, config = tf.keras.layers.ZeroPadding2D().get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["padding"] =[int(9/5*(seed[0] % 5))+1 for _ in range(2)]
    
    return tf.keras.layers.ZeroPadding2D.from_config(config)

def addAverPool1D(name, config = tf.keras.layers.AveragePooling1D(pool_size=1).get_config(), seed = [0,0]):
    config["name"] = f'layer_{name}'
    config["strides"] = int(5/5*(seed[0] % 5))+1
    config["padding"] = ['valid', 'same'][seed[1] % 2]

    return tf.keras.layers.AveragePooling1D.from_config(config)

def addAverPool2D(name, config = tf.keras.layers.AveragePooling2D(pool_size=(1,1)).get_config(), seed = [0,0]):
    config["name"] = f'layer_{name}'
    config["strides"] = [int(5/5*(seed[0] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[1] % 2]

    return tf.keras.layers.AveragePooling2D.from_config(config)

def addAverPool3D(name, config = tf.keras.layers.AveragePooling3D(pool_size=(1,1,1)).get_config(), seed = [0,0]):
    config["name"] = f'layer_{name}'
    config["strides"] = [int(5/5*(seed[0] % 5))+1 for _ in range(3)]
    config["padding"] = ['valid', 'same'][seed[1] % 2]

    return tf.keras.layers.AveragePooling3D.from_config(config)


def addConv1D(name, config = tf.keras.layers.Conv1D(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = int(5/5*(seed[1] % 5))+1
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'][seed[3] % 11]
    
    return tf.keras.layers.Conv1D.from_config(config)

def addConv3D(name, config = tf.keras.layers.Conv3D(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(3)]
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'][seed[3] % 11]
    
    return tf.keras.layers.Conv3D.from_config(config)

def addLeakyReLU(name, config = tf.keras.layers.LeakyReLU().get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["alpha"] = 1/5*(seed % 5)
    
    return tf.keras.layers.LeakyReLU.from_config(config)

def addLocallyConnected1D(name, config = tf.keras.layers.LocallyConnected1D(filters=1, kernel_size=1).get_config(), seed = [0,0,0]):
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'][seed[2] % 11]
    
    return tf.keras.layers.LocallyConnected1D.from_config(config)

def addLocallyConnected2D(name, config = tf.keras.layers.LocallyConnected2D(filters=1, kernel_size=1).get_config(), seed = [0,0,0]):
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(2)]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'][seed[2] % 11]
    
    return tf.keras.layers.LocallyConnected2D.from_config(config)

def addReLU(name, config = tf.keras.layers.ReLU().get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["max_value"] = 1/5*(seed[0] % 5)
    
    return tf.keras.layers.ReLU.from_config(config)

def addRepeatVector(name, config = tf.keras.layers.RepeatVector(n=2).get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["n"] = int(1000/5*(seed[0] % 5))
    
    return tf.keras.layers.RepeatVector.from_config(config)

def addSeparableConv1D(name, config = tf.keras.layers.SeparableConv1D(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1]
    config["padding"] = ['valid', 'same', 'causal'][seed[2] % 3]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[3] % 11]
    
    return tf.keras.layers.SeparableConv1D.from_config(config)

def addSeparableConv2D(name, config = tf.keras.layers.SeparableConv2D(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[3] % 11]
    
    return tf.keras.layers.SeparableConv2D.from_config(config)

def addSimpleRNN(name, config = tf.keras.layers.SimpleRNN(units=2).get_config(), seed = [0,0,0,0,0,0,0,0]):
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0] % 5))
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[1] % 11]
    config["use_bias"] = [True, False][seed[2] % 2]
    config["dropout"] = 1/5*(seed[3] % 5)
    config["recurrent_dropout"] = 1/5*(seed[4] % 5)
    config["return_sequences"] = [True, False][seed[5] % 2]
    config["return_state"] = [True, False][seed[6] % 2]
    config["stateful"] = [True, False][seed[7] % 2]

    return tf.keras.layers.SimpleRNN.from_config(config)

def addSimpleRNNCell(name, config = tf.keras.layers.SimpleRNNCell(units=2).get_config(), seed = [0,0,0,0,0]):
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0] % 5))
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[1] % 11]
    config["use_bias"] = [True, False][seed[2] % 2]
    config["dropout"] = 1/5*(seed[3] % 5)
    config["recurrent_dropout"] = 1/5*(seed[4] % 5)

    return tf.keras.layers.SimpleRNNCell.from_config(config)

def addSpatialDropout1D(name, config = tf.keras.layers.SpatialDropout1D(rate=1).get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["rate"] = int(1/5*(seed[0] % 5))
    
    return tf.keras.layers.SpatialDropout1D.from_config(config)

def addSpatialDropout2D(name, config = tf.keras.layers.SpatialDropout2D(rate=1).get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["rate"] = int(1/5*(seed[0] % 5))

    return tf.keras.layers.SpatialDropout2D.from_config(config)   

def addSpatialDropout3D(name, config = tf.keras.layers.SpatialDropout3D(rate=1).get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["rate"] = int(1/5*(seed[0] % 5))

    return tf.keras.layers.SpatialDropout3D.from_config(config)

def addThresholdedReLU(name, config = tf.keras.layers.ThresholdedReLU().get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["theta"] = 1/5*(seed[0] % 5)
    
    return tf.keras.layers.ThresholdedReLU.from_config(config)

def addConv2DTranspose(name, config = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[3] % 11]
    
    return tf.keras.layers.Conv2DTranspose.from_config(config)

def addConv3DTranspose(name, config = tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0]  % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(3)]
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[3] % 11]
    
    return tf.keras.layers.Conv3DTranspose.from_config(config)

def addDepthwiseConv2D(name, config = tf.keras.layers.DepthwiseConv2D(kernel_size=1).get_config(), seed = [0,0,0]):
    config["name"] = f'layer_{name}'
    config["strides"] = [int(5/5*(seed[0] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[1] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[2] % 11]
    
    return tf.keras.layers.DepthwiseConv2D.from_config(config)

def addFlatten(name, config = tf.keras.layers.Flatten().get_config(), seed = []):
    config["name"] = f'layer_{name}'

    return tf.keras.layers.Flatten.from_config(config)

def addGRU(name, config = tf.keras.layers.GRU(units=2).get_config(), seed = [0,0,0]):
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0] % 5))
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[1] % 11]
    config["use_bias"] = [True, False][seed[2] % 2]

    return tf.keras.layers.GRU.from_config(config)

def addGRUCell(name, config = tf.keras.layers.GRUCell(units=2).get_config(), seed = [0,0,0]):
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0]  % 5))
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[1] % 11]
    config["use_bias"] = [True, False][seed[2] % 2]
    
    return tf.keras.layers.GRUCell.from_config(config)

def addGaussianDropout(name, config = tf.keras.layers.GaussianDropout(rate=0.1).get_config(), seed = [0,0]):
    config["name"] = f'layer_{name}'
    config["rate"] = 1/5*(seed[0] % 5)
    config["seed"] = seed[1] % 5
    
    return tf.keras.layers.GaussianDropout.from_config(config)

def addGaussianNoise(name, config = tf.keras.layers.GaussianNoise(stddev=0.1).get_config(), seed = [0,0]):
    config["name"] = f'layer_{name}'
    config["stddev"] = 1/5*(seed[0] % 5)
    config["seed"] = seed[1] % 5
    
    return tf.keras.layers.GaussianNoise.from_config(config)

def addGloAverPool1D(name, config = tf.keras.layers.GlobalAveragePooling1D().get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["keepdims"] = [True, False][seed[0] % 2]

    return tf.keras.layers.GlobalAveragePooling1D.from_config(config)


def addGloAverPool2D(name, config = tf.keras.layers.GlobalAveragePooling2D().get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["keepdims"] = [True, False][seed[0] % 2]

    return tf.keras.layers.GlobalAveragePooling2D.from_config(config)

def addGloAverPool3D(name, config = tf.keras.layers.GlobalAveragePooling3D().get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["keepdims"] = [True, False][seed[0] % 2]

    return tf.keras.layers.GlobalAveragePooling3D.from_config(config)

def addGloMaxPool1D(name, config = tf.keras.layers.GlobalMaxPool1D().get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["keepdims"] = [True, False][seed[0] % 2]

    return tf.keras.layers.GlobalMaxPool1D.from_config(config)


def addGloMaxPool2D(name, config = tf.keras.layers.GlobalMaxPooling2D().get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["keepdims"] = [True, False][seed[0] % 2]

    return tf.keras.layers.GlobalMaxPooling2D.from_config(config)

def addGloMaxPool3D(name, config = tf.keras.layers.GlobalMaxPooling3D().get_config(), seed = [0]):
    config["name"] = f'layer_{name}'
    config["keepdims"] = [True, False][seed[0] % 2]

    return tf.keras.layers.GlobalMaxPooling3D.from_config(config)


def addLSTM(name, config = tf.keras.layers.LSTM(units=2).get_config(), seed = [0,0,0,0,0,0,0,0,0,0,0]):
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0] % 5))
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[1] % 11]
    config["use_bias"] = [True, False][seed[2] % 2]
    config["dropout"] = 1/5*(seed[3] % 5)
    config["recurrent_dropout"] = 1/5*(seed[4] % 5)
    config["return_sequences"] = [True, False][seed[5] % 2]
    config["return_state"] = [True, False][seed[6] % 2]
    config["go_backwards"] = [True, False][seed[7] % 2]
    config["stateful"] = [True, False][seed[8] % 2]    
    config["time_major"] = [True, False][seed[9] % 2]
    config["unroll"] = [True, False][seed[10] % 2]

    return tf.keras.layers.LSTM.from_config(config)

def addLSTMCell(name, config = tf.keras.layers.LSTMCell(units=2).get_config(), seed = [0,0,0,0,0]):
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0] % 5))
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[1] % 11]
    config["use_bias"] = [True, False][seed[2] % 2]
    config["dropout"] = 1/5*(seed[3] % 5)
    config["recurrent_dropout"] = 1/5*(seed[4] % 5)

    return tf.keras.layers.LSTMCell.from_config(config)

def addLayerNormal(name, config = tf.keras.layers.LayerNormalization().get_config(), seed = [0,0,0]):
    config["name"] = f'layer_{name}'
    config["epsilon"] = 1/5*(seed[0] % 5)
    config["center"] = [True, False][seed[1] % 2]
    config["scale"] = [True, False][seed[2] % 2]
    
    return tf.keras.layers.LayerNormalization.from_config(config)

def addReshape(model, target_shape):
    cnt = 1
    size = []
    for _ in range(1, len(model.layers[-1].output_shape)):
        cnt *= model.layers[-1].output_shape[_]
    size.append(cnt)
    for _ in range(target_shape-1):
        size.append(1)
        
    return tf.keras.layers.Reshape(size)

def extractModelArchitect(model):
    model_architect = []
    for layer_config in model.get_config()['layers']:
        model_architect.append(layer_config['class_name'])
    return model_architect

def modelReduction(architect):
    reduced_architect = []
    for layer in architect:
        if layer not in reduced_architect:
            reduced_architect.append(layer)
    return reduced_architect

def buildModelByArtchitect(architect, input_shape, seed):
    model = tf.keras.Sequential()
    layer_count = 0
    two_dim_layer = []
    three_dim_layer = ['GlobalAveragePooling2D']
    four_dim_layer = []
    for layer_name in architect:
        if (layer_name == 'InputLayer'):
            model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        elif (layer_name == 'Dense'): #Dense
            model.add(addDense(layer_count, seed=seed))
        elif (layer_name == 'Conv2D'): #Conv2D
            model.add(addConv2D(layer_count, seed=seed))
        elif (layer_name == 'BatchNormalization'): #BatchNormalization
            model.add(addBatchNorm(layer_count, seed=seed))
        elif (layer_name == 'MaxPooling2D'): #MaxPooling2D
            model.add(addMaxPool(layer_count, seed=seed))
        elif (layer_name == 'Dropout'): #Dropout
            model.add(addDropout(layer_count, seed=seed))
        elif (layer_name == 'Activation'): #Activation
            model.add(addActivation(layer_count, seed=seed))
        elif (layer_name == 'ZeroPadding2D'): #ZeroPadding2D
            model.add(addZeroPadd(layer_count, seed=seed))
        elif (layer_name == 'AveragePooling1D'): #AveragePooling1D
            model.add(addAverPool1D(layer_count, seed=seed))
        elif (layer_name == 'AveragePooling2D'): #AveragePooling2D
            model.add(addAverPool2D(layer_count, seed=seed))
        elif (layer_name == 'AveragePooling3D'): #AveragePooling3D
            model.add(addAverPool3D(layer_count, seed=seed))
        elif (layer_name == 'Conv1D'): #Conv1D
            model.add(addConv1D(layer_count, seed=seed))
        elif (layer_name == 'Conv3D'): #Conv3D
            model.add(addConv3D(layer_count, seed=seed))
        elif (layer_name == 'LeakyReLU'): #LeakyReLU
            model.add(addLeakyReLU(layer_count, seed=seed))
        elif (layer_name == 'LocallyConnected1D'): #LocallyConnected1D
            model.add(addLocallyConnected1D(layer_count, seed=seed))
        elif (layer_name == 'LocallyConnected2D'): #LocallyConnected2D
            model.add(addLocallyConnected2D(layer_count, seed=seed))
        elif (layer_name == 'ReLU'): #ReLU
            model.add(addReLU(layer_count, seed=seed))
        elif (layer_name == 'RepeatVector'): #RepeatVector
            model.add(addRepeatVector(layer_count, seed=seed))
        elif (layer_name == 'SeparableConv1D'): #SeparableConv1D
            model.add(addSeparableConv1D(layer_count, seed=seed))
        elif (layer_name == 'SeparableConv2D'): #SeparableConv2D
            model.add(addSeparableConv2D(layer_count, seed=seed))
        elif (layer_name == 'SimpleRNN'): #SimpleRNN
            model.add(addSimpleRNN(layer_count, seed=seed))
        elif (layer_name == 'SimpleRNNCell'): #SimpleRNNCell
            model.add(addSimpleRNNCell(layer_count, seed=seed))
        elif (layer_name == 'SpatialDropout1D'): #SpatialDropout1D
            model.add(addSpatialDropout1D(layer_count, seed=seed))
        elif (layer_name == 'SpatialDropout2D'): #SpatialDropout2D
            model.add(addSpatialDropout2D(layer_count, seed=seed))
        elif (layer_name == 'SpatialDropout3D'): #SpatialDropout3D
            model.add(addSpatialDropout3D(layer_count, seed=seed))
        elif (layer_name == 'ThresholdedReLU'): #ThresholdedReLU
            model.add(addThresholdedReLU(layer_count, seed=seed))
        elif (layer_name == 'Conv2DTranspose'): #Conv2DTranspose
            model.add(addConv2DTranspose(layer_count, seed=seed))
        elif (layer_name == 'Conv3DTranspose'): #Conv3DTranspose
            model.add(addConv3DTranspose(layer_count, seed=seed))
        elif (layer_name == 'DepthwiseConv2D'): #DepthwiseConv2D
            model.add(addDepthwiseConv2D(layer_count, seed=seed))
        elif (layer_name == 'Flatten'): #Flatten
            model.add(addFlatten(layer_count, seed=seed))
        elif (layer_name == 'GRU'): #GRU
            model.add(addGRU(layer_count, seed=seed))
        elif (layer_name == 'GRUCell'): #GRUCell
            model.add(addGRUCell(layer_count, seed=seed))
        elif (layer_name == 'GaussianDropout'): #GaussianDropout
            model.add(addGaussianDropout(layer_count, seed=seed))
        elif (layer_name == 'GaussianNoise'): #GaussianNoise
            model.add(addGaussianNoise(layer_count, seed=seed))
        elif (layer_name == 'GlobalAveragePooling1D'): #GlobalAveragePooling1D
            model.add(addGloAverPool1D(layer_count, seed=seed))
        elif (layer_name == 'GlobalAveragePooling2D'): #GlobalAveragePooling2D
            model.add(addGloAverPool2D(layer_count, seed=seed))
        elif (layer_name == 'GlobalAveragePooling3D'): #GlobalAveragePooling3D
            model.add(addGloAverPool3D(layer_count, seed=seed))
        elif (layer_name == 'GlobalMaxPooling1D'): #GlobalMaxPooling1D
            model.add(addGloMaxPool1D(layer_count, seed=seed))
        elif (layer_name == 'GlobalMaxPooling2D'): #GlobalMaxPooling2D
            model.add(addGloMaxPool2D(layer_count, seed=seed))
        elif (layer_name == 'GlobalMaxPooling3D'): #GlobalMaxPooling3D
            model.add(addGloMaxPool3D(layer_count, seed=seed))
        elif (layer_name == 'LSTM'): #LSTM
            model.add(addLSTM(layer_count, seed=seed))
        elif (layer_name == 'LSTMCell'): #LSTMCell
            model.add(addLSTMCell(layer_count, seed=seed))
        elif (layer_name == 'LayerNormalization'): #LayerNormalization
            model.add(addLayerNormal(layer_count, seed=seed))
        elif (layer_name == 'Reshape'):
            following_layer = architect[layer_count+1]
            if (following_layer in two_dim_layer):
                model.add(addReshape(model, 2))
            elif (following_layer in three_dim_layer):
                model.add(addReshape(model, 3))
            elif (following_layer in four_dim_layer):
                model.add(addReshape(model, 4))
        else:
            print(f'unidentify layer: {layer_name}')
        layer_count += 1
    return model

def get_paras_num(layer_name):
    d = {
        'Dense' : 1,
        'Conv2D' : 4,
        'BatchNormalization' : 2,
        'MaxPooling2D' : 2,
        'Dropout' : 2,
        'Activation' : 1,
        'ZeroPadding2D' : 1,
        'AveragePooling1D' : 2,
        'AveragePooling2D' : 2,
        'AveragePooling3D' : 2,
        'Conv1D' : 4,
        'Conv3D' : 4,
        'LeakyReLU' : 1,
        'LocallyConnected1D' : 3,
        'LocallyConnected2D' : 3,
        'ReLU' : 1,
        'RepeatVector' : 1,
        'SeparableConv1D' : 4,
        'SeparableConv2D' : 4,
        'SimpleRNN' : 8,
        'SimpleRNNCell' : 5,
        'SpatialDropout1D' : 1,
        'SpatialDropout2D' : 1,
        'SpatialDropout3D' : 1,
        'ThresholdedReLU' : 1,
        'Conv2DTranspose' : 4,
        'Conv3DTranspose' : 4,
        'DepthwiseConv2D' : 3,
        'Flatten' : 0,
        'GRU' : 3,
        'GRUCell' : 3,
        'GaussianDropout' : 2,
        'GaussianNoise' : 2,
        'GlobalAveragePooling1D' : 1,
        'GlobalAveragePooling2D' : 1,
        'GlobalAveragePooling3D' : 1,
        'GlobalMaxPooling1D' : 1,
        'GlobalMaxPooling2D' : 1,
        'GlobalMaxPooling3D' : 1,
        'LSTM' : 11,
        'LSTMCell' : 5,
        'LayerNormalization' : 3
    }
    return d[layer_name]

import keras
import pandas as pd
import numpy as np

def addDense(name, config = keras.layers.Dense(units=1, input_shape=[1]).get_config(), seed = [0]):
    if seed is None:
        seed = [0]
    config["name"] = f'layer_{name}'
    config["units"] = int(500/5*(seed[0] % 5 + 1))
    return keras.layers.Dense.from_config(config)

def addConv2D(name, config = keras.layers.Conv2D(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    if seed is None:
        seed = [0,0,0,0]
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid','exponential', 'linear'][seed[3] % 11]
    
    return keras.layers.Conv2D.from_config(config)
    
def addBatchNorm(name, config = keras.layers.BatchNormalization().get_config(), seed = [0,0]):
    if seed is None:
        seed = [0,0]
    config["name"] = f'layer_{name}'
    config["momentum"] = 1/5*(seed[0] % 5)
    config["epsilon"] = 1/5*(seed[1] % 5)
    
    return keras.layers.BatchNormalization.from_config(config)
    
    
def addMaxPool(name, config = keras.layers.MaxPooling2D(pool_size=(1,1)).get_config(), seed = [0,0]):
    if seed is None:
        seed = [0,0]
    config["name"] = f'layer_{name}'
    config["strides"] = [int(5/5*(seed[0] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[1] % 2]
    
    return keras.layers.MaxPooling2D.from_config(config)
    

def addDropout(name, config = keras.layers.Dropout(rate=0.1).get_config(), seed = [0,0]):
    if seed is None:
        seed = [0,0]
    config["name"] = f'layer_{name}'
    config["rate"] = 1/5*(seed[0] % 5)
    config["seed"] = seed[1] % 5
    
    return keras.layers.Dropout.from_config(config)

def addActivation(name, config = keras.layers.Activation(activation="softmax").get_config(), seed = [0]):
    if seed is None:
        seed = [0,0]
    config["name"] = f'layer_{name}'
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'][seed[0] % 11]
    
    return keras.layers.Activation.from_config(config)

def addZeroPadd(name, config = keras.layers.ZeroPadding2D().get_config(), seed = [0]):
    if seed is None:
        seed = [0]
    config["name"] = f'layer_{name}'
    config["padding"] =[int(9/5*(seed[0] % 5))+1 for _ in range(2)]
    
    return keras.layers.ZeroPadding2D.from_config(config)

def addAverPool1D(name, config = keras.layers.AveragePooling1D(pool_size=1).get_config(), seed = [0,0]):
    if seed is None:
        seed = [0,0]
    config["name"] = f'layer_{name}'
    config["strides"] = int(5/5*(seed[0] % 5))+1
    config["padding"] = ['valid', 'same'][seed[1] % 2]

    return keras.layers.AveragePooling1D.from_config(config)

def addAverPool2D(name, config = keras.layers.AveragePooling2D(pool_size=(1,1)).get_config(), seed = [0,0]):
    if seed is None:
        seed = [0,0]
    config["name"] = f'layer_{name}'
    config["strides"] = [int(5/5*(seed[0] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[1] % 2]

    return keras.layers.AveragePooling2D.from_config(config)

def addAverPool3D(name, config = keras.layers.AveragePooling3D(pool_size=(1,1,1)).get_config(), seed = [0,0]):
    if seed is None:
        seed = [0,0]
    config["name"] = f'layer_{name}'
    config["strides"] = [int(5/5*(seed[0] % 5))+1 for _ in range(3)]
    config["padding"] = ['valid', 'same'][seed[1] % 2]

    return keras.layers.AveragePooling3D.from_config(config)


def addConv1D(name, config = keras.layers.Conv1D(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    if seed is None:
        seed = [0,0,0,0]
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = int(5/5*(seed[1] % 5))+1
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'][seed[3] % 11]
    
    return keras.layers.Conv1D.from_config(config)

def addConv3D(name, config = keras.layers.Conv3D(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    if seed is None:
        seed = [0,0,0,0]
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(3)]
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'][seed[3] % 11]
    
    return keras.layers.Conv3D.from_config(config)

def addLeakyReLU(name, config = keras.layers.LeakyReLU().get_config(), seed = []):
    if seed is None:
        seed = []
    config["name"] = f'layer_{name}'
#     config["alpha"] = 1/5*(seed % 5)
    
    return keras.layers.LeakyReLU.from_config(config)

def addLocallyConnected1D(name, config = keras.layers.LocallyConnected1D(filters=1, kernel_size=1).get_config(), seed = [0,0,0]):
    if seed is None:
        seed = [0,0,0]
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'][seed[2] % 11]
    
    return keras.layers.LocallyConnected1D.from_config(config)

def addLocallyConnected2D(name, config = keras.layers.LocallyConnected2D(filters=1, kernel_size=1).get_config(), seed = [0,0,0]):
    if seed is None:
        seed = [0,0,0]
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(2)]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'][seed[2] % 11]
    
    return keras.layers.LocallyConnected2D.from_config(config)

def addReLU(name, config = keras.layers.ReLU().get_config(), seed = [0]):
    if seed is None:
        seed = [0]
    config["name"] = f'layer_{name}'
    config["max_value"] = 1/5*(seed[0] % 5)
    
    return keras.layers.ReLU.from_config(config)

def addRepeatVector(name, config = keras.layers.RepeatVector(n=2).get_config(), seed = [0]):
    if seed is None:
        seed = [0]
    config["name"] = f'layer_{name}'
    config["n"] = int(1000/5*(seed[0] % 5))
    
    return keras.layers.RepeatVector.from_config(config)

def addSeparableConv1D(name, config = keras.layers.SeparableConv1D(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    if seed is None:
        seed = [0,0,0,0]
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1]
    config["padding"] = ['valid', 'same', 'causal'][seed[2] % 3]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[3] % 11]
    
    return keras.layers.SeparableConv1D.from_config(config)

def addSeparableConv2D(name, config = keras.layers.SeparableConv2D(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    if seed is None:
        seed = [0,0,0,0]
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[3] % 11]
    
    return keras.layers.SeparableConv2D.from_config(config)

def addSimpleRNN(name, config = keras.layers.SimpleRNN(units=2).get_config(), seed = [0,0,0,0,0,0]):
    if seed is None:
        seed = [0,0,0,0,0,0]
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0] % 5))
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[1] % 11]
    config["use_bias"] = [True, False][seed[2] % 2]
    config["dropout"] = 1/5*(seed[3] % 5)
    config["recurrent_dropout"] = 1/5*(seed[4] % 5)
    config["return_sequences"] = [True, False][seed[5] % 2]
#     config["return_state"] = [True, False][seed[6] % 2]
#     config["stateful"] = [True, False][seed[7] % 2]

    return keras.layers.SimpleRNN.from_config(config)

# def addSimpleRNNCell(name, config = keras.layers.SimpleRNNCell(units=2).get_config(), seed = [0,0,0,0,0]):
#     if seed is None:
#         seed = [0,0,0,0,0]
#     config["name"] = f'layer_{name}'
#     config["units"] = int(1000/5*(seed[0] % 5))
#     config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
#                             'exponential', 'linear'][seed[1] % 11]
#     config["use_bias"] = [True, False][seed[2] % 2]
#     config["dropout"] = 1/5*(seed[3] % 5)
#     config["recurrent_dropout"] = 1/5*(seed[4] % 5)

#     return keras.layers.SimpleRNNCell.from_config(config)

def addSpatialDropout1D(name, config = keras.layers.SpatialDropout1D(rate=1).get_config(), seed = [0]):
    if seed is None:
        seed = [0]
    config["name"] = f'layer_{name}'
    config["rate"] = int(1/5*(seed[0] % 5))
    
    return keras.layers.SpatialDropout1D.from_config(config)

def addSpatialDropout2D(name, config = keras.layers.SpatialDropout2D(rate=1).get_config(), seed = [0]):
    if seed is None:
        seed = [0]
    config["name"] = f'layer_{name}'
    config["rate"] = int(1/5*(seed[0] % 5))

    return keras.layers.SpatialDropout2D.from_config(config)   

def addSpatialDropout3D(name, config = keras.layers.SpatialDropout3D(rate=1).get_config(), seed = [0]):
    if seed is None:
        seed = [0]
    config["name"] = f'layer_{name}'
    config["rate"] = int(1/5*(seed[0] % 5))

    return keras.layers.SpatialDropout3D.from_config(config)

def addThresholdedReLU(name, config = keras.layers.ThresholdedReLU().get_config(), seed = [0]):
    if seed is None:
        seed = [0]
    config["name"] = f'layer_{name}'
    config["theta"] = 1/5*(seed[0] % 5)
    
    return keras.layers.ThresholdedReLU.from_config(config)

def addConv2DTranspose(name, config = keras.layers.Conv2DTranspose(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    if seed is None:
        seed = [0,0,0,0]
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0] % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[3] % 11]
    
    return keras.layers.Conv2DTranspose.from_config(config)

def addConv3DTranspose(name, config = keras.layers.Conv3DTranspose(filters=1, kernel_size=1).get_config(), seed = [0,0,0,0]):
    if seed is None:
        seed = [0,0,0,0]
    config["name"] = f'layer_{name}'
    config["filters"] = int(20/5*(seed[0]  % 5))+1
    config["strides"] = [int(5/5*(seed[1] % 5))+1 for _ in range(3)]
    config["padding"] = ['valid', 'same'][seed[2] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[3] % 11]
    
    return keras.layers.Conv3DTranspose.from_config(config)

def addDepthwiseConv2D(name, config = keras.layers.DepthwiseConv2D(kernel_size=1).get_config(), seed = [0,0,0]):
    if seed is None:
        seed = [0,0,0]
    config["name"] = f'layer_{name}'
    config["strides"] = [int(5/5*(seed[0] % 5))+1 for _ in range(2)]
    config["padding"] = ['valid', 'same'][seed[1] % 2]
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[2] % 11]
    
    return keras.layers.DepthwiseConv2D.from_config(config)

def addFlatten(name, config = keras.layers.Flatten().get_config(), seed = []):
    if seed is None:
        seed = []
    config["name"] = f'layer_{name}'

    return keras.layers.Flatten.from_config(config)

def addGRU(name, config = keras.layers.GRU(units=2).get_config(), seed = [0,0,0]):
    if seed is None:
        seed = [0,0,0]
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0] % 5))
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[1] % 11]
    config["use_bias"] = [True, False][seed[2] % 2]

    return keras.layers.GRU.from_config(config)

def addGRUCell(name, config = keras.layers.GRUCell(units=2).get_config(), seed = [0,0,0]):
    if seed is None:
        seed = [0,0,0]
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0]  % 5))
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[1] % 11]
    config["use_bias"] = [True, False][seed[2] % 2]
    
    return keras.layers.GRUCell.from_config(config)

def addGaussianDropout(name, config = keras.layers.GaussianDropout(rate=0.1).get_config(), seed = [0]):
    if seed is None:
        seed = [0]
    config["name"] = f'layer_{name}'
    config["rate"] = 1/5*(seed[0] % 5)
#     config["seed"] = seed[1] % 5
    
    return keras.layers.GaussianDropout.from_config(config)

def addGaussianNoise(name, config = keras.layers.GaussianNoise(stddev=0.1).get_config(), seed = [0]):
    if seed is None:
        seed = [0]
    config["name"] = f'layer_{name}'
    config["stddev"] = 1/5*(seed[0] % 5)
#     config["seed"] = seed[1] % 5
    
    return keras.layers.GaussianNoise.from_config(config)

def addGloAverPool1D(name, config = keras.layers.GlobalAveragePooling1D().get_config(), seed = []):
    if seed is None:
        seed = []
    config["name"] = f'layer_{name}'

    return keras.layers.GlobalAveragePooling1D.from_config(config)


def addGloAverPool2D(name, config = keras.layers.GlobalAveragePooling2D().get_config(), seed = []):
    if seed is None:
        seed = []
    config["name"] = f'layer_{name}'

    return keras.layers.GlobalAveragePooling2D.from_config(config)

def addGloAverPool3D(name, config = keras.layers.GlobalAveragePooling3D().get_config(), seed = []):
    if seed is None:
        seed = []
    config["name"] = f'layer_{name}'

    return keras.layers.GlobalAveragePooling3D.from_config(config)

def addGloMaxPool1D(name, config = keras.layers.GlobalMaxPool1D().get_config(), seed = []):
    if seed is None:
        seed = []
    config["name"] = f'layer_{name}'

    return keras.layers.GlobalMaxPool1D.from_config(config)


def addGloMaxPool2D(name, config = keras.layers.GlobalMaxPooling2D().get_config(), seed = []):
    if seed is None:
        seed = []
    config["name"] = f'layer_{name}'

    return keras.layers.GlobalMaxPooling2D.from_config(config)

def addGloMaxPool3D(name, config = keras.layers.GlobalMaxPooling3D().get_config(), seed = []):
    if seed is None:
        seed = []
    config["name"] = f'layer_{name}'

    return keras.layers.GlobalMaxPooling3D.from_config(config)


def addLSTM(name, config = keras.layers.LSTM(units=2).get_config(), seed = [0,0,0,0,0,0,0,0]):
    if seed is None:
        seed = [0,0,0,0,0,0,0,0]
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0] % 5))
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[1] % 11]
    config["use_bias"] = [True, False][seed[2] % 2]
    config["dropout"] = 1/5*(seed[3] % 5)
    config["recurrent_dropout"] = 1/5*(seed[4] % 5)
    config["return_sequences"] = [True, False][seed[5] % 2]
#     config["return_state"] = [True, False][seed[6] % 2]
    config["go_backwards"] = [True, False][seed[6] % 2]
#     config["stateful"] = [True, False][seed[7] % 2]    
    config["time_major"] = [True, False][seed[7] % 2]
#     config["unroll"] = [True, False][seed[10] % 2]

    return keras.layers.LSTM.from_config(config)

def addLSTMCell(name, config = keras.layers.LSTMCell(units=2).get_config(), seed = [0,0,0,0,0]):
    if seed is None:
        seed = [0,0,0,0,0]
    config["name"] = f'layer_{name}'
    config["units"] = int(1000/5*(seed[0] % 5))
    config["activation"] = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                            'exponential', 'linear'][seed[1] % 11]
    config["use_bias"] = [True, False][seed[2] % 2]
    config["dropout"] = 1/5*(seed[3] % 5)
    config["recurrent_dropout"] = 1/5*(seed[4] % 5)

    return keras.layers.LSTMCell.from_config(config)

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

def buildModelByArtchitect(architect, seedlist, input_shape, layer_df=pd.read_csv('layer_dict.csv', index_col=0)):
    assert len(architect) == len(seedlist), 'length of layers not equal seedlist'
    model = keras.Sequential([keras.layers.InputLayer(input_shape=input_shape)])
    
    
    for i, layer_name in enumerate(architect):
        aval_input = eval(layer_df.loc[layer_name]['input_shape'])
        add_function= eval(layer_df.loc[layer_name]['add_function'])
        prev_output_shape = len(input_shape) if i == 0 else (len(model.layers[-1].output_shape)-1) 
        
        if prev_output_shape in aval_input:
            model.add(add_function(i, seed=seedlist[i]))
        else:
            #calculate output layer shape
            prev_output = input_shape if i == 0 else model.layers[-1].output_shape[1:]
            reshape_shape = [np.prod(prev_output)]
            for _ in range(aval_input[0] - 1): reshape_shape.append(1)
            model.add(keras.layers.Reshape(reshape_shape))
            model.add(add_function(i, seed=seedlist[i]))
            
    return model
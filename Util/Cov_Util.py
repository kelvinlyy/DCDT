tensorflow_root = r"/data/ykchanbf/anaconda3/envs/fyp/lib/python3.6/site-packages/tensorflow*"
keras_root = r"/data/ykchanbf/anaconda3/envs/fyp/lib/python3.6/site-packages/keras*"


import coverage
import keras
import pandas as pd
from re import match

from Util.Layer_Util import addActivation, addAverPool1D, addAverPool2D, addAverPool3D, addBatchNorm, addConv1D, addConv2D, addConv2DTranspose, addConv3D, addConv3DTranspose, addDense, addDepthwiseConv2D, addDropout, addFlatten, addGRU, addGaussianDropout, addGaussianNoise, addGloAverPool1D, addGloAverPool2D, addGloAverPool3D, addGloMaxPool1D, addGloMaxPool2D, addGloMaxPool3D, addLSTM, addLeakyReLU, addLocallyConnected1D, addLocallyConnected2D, addMaxPool, addReLU, addRepeatVector, addSeparableConv1D, addSeparableConv2D, addSimpleRNN, addSpatialDropout1D, addSpatialDropout2D, addSpatialDropout3D, addThresholdedReLU, addZeroPadd, buildModelByArtchitect, extractModelArchitect, modelReduction

def lines2Vector(lines):
    lineVector = []
    if lines:
        n = 1
        m = 0
        while (n <= lines[-1]):
            if lines[m] == n:
                lineVector.append(1)
                m += 1
            else:
                lineVector.append(0)
            n += 1
    return lineVector

#old version
# def calc_coverage(model, test_data):
#     cov = coverage.Coverage()
#     cov.start()
    
#     predictions = model.predict(test_data)
#     cov.stop()    
#     covData = cov.get_data()
#     covList = [(files, sorted(covData.lines(files))) for files in covData.measured_files()]      
    
#     return covList

def calc_coverage(model_architecture, seedlist, input_data):
    cov = coverage.Coverage()
    cov.start()
    
    
    if len(input_data.shape) == 3:
        input_data = input_data[None,:]
    model = buildModelByArtchitect(model_architecture, seedlist, input_data.shape[1:])                           
    predictions = model.predict(input_data)
    
    cov.stop()    
    covData = cov.get_data()
    covList = [(files, sorted(covData.lines(files))) for files in covData.measured_files()]      
    
    return covList
 

def total_lines(covList, root=tensorflow_root):
    cnt = 0
    for _ in covList:
        if match(root, _[0]) or match(keras_root, _[0]): 
            cnt += len(_[1])
    return cnt

def delta_lines(covList1, covList2, root=tensorflow_root):
    cnt = 0
    for files in range(len(covList1)):
        
        if match(root, covList1[files][0]) or match(keras_root, covList1[files][0]):
            a = set(covList1[files][1])
            b = set(covList2[files][1])
            cnt += len(a.difference(b))
            cnt += len(b.difference(a))    
        
    return cnt
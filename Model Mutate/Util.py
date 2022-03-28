import coverage
from tensorflow import keras
import pandas as pd

from Layer_Util import addActivation, addAverPool1D, addAverPool2D, addAverPool3D, addBatchNorm, addConv1D, addConv2D, addConv2DTranspose, addConv3D, addConv3DTranspose, addDense, addDepthwiseConv2D, addDropout, addFlatten, addGRU, addGRUCell, addGaussianDropout, addGaussianNoise, addGloAverPool1D, addGloAverPool2D, addGloAverPool3D, addGloMaxPool1D, addGloMaxPool2D, addGloMaxPool3D, addLSTM, addLSTMCell, addLayerNormal, addLeakyReLU, addLocallyConnected1D, addLocallyConnected2D, addMaxPool, addReLU, addRepeatVector, addReshape, addSeparableConv1D, addSeparableConv2D, addSimpleRNN, addSimpleRNNCell, addSpatialDropout1D, addSpatialDropout2D, addSpatialDropout3D, addThresholdedReLU, addZeroPadd, buildModelByArtchitect, extractModelArchitect, get_paras_num, modelReduction

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

def calc_coverage(model, test_data):
    cov = coverage.Coverage()
    cov.start()
    
    predictions = model.predict(test_data)
    cov.stop()    
    covData = cov.get_data()
    covList = [(files, sorted(covData.lines(files))) for files in covData.measured_files()]      
    
    return covList
 

def total_lines(covList):
    cnt = 0
    for _ in covList:
        cnt += len(_[1])
    return cnt

def delta_lines(covList1, covList2):
    cnt = 0
    for files in range(len(covList1)):
        a = set(covList1[files][1])
        b = set(covList2[files][1])
        cnt += len(a.difference(b))
        cnt += len(b.difference(a))    
        
    return cnt
# Samples for testing... =====================================================================
import numpy as np
import matplotlib.pyplot as plt

# def loadSphere2():
#     # Import sampleGUV1 - the 3-dimensional array that holds the fluorescence density for each voxel.'''
#     # With GUV1, 15x15x15 ranges from 15 - 37
#     with open('samples/SolidSphere2Trimmed.txt', 'r') as f: text = f.read()
#     for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
#     array = eval(text)
#     boundingBoxDim = [5.1899999999999995, 5.1899999999999995, 5.1899999999999995]
#     return np.array(array)
#
# def loadSphere1():
#     # Import sampleGUV1 - the 3-dimensional array that holds the fluorescence density for each voxel.'''
#     # With GUV1, 15x15x15 ranges from 15 - 37
#     with open('samples/SolidSphere1Trimmed.txt', 'r') as f: text = f.read()
#     for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
#     array = eval(text)
#     boundingBoxDim = [8.65, 8.65, 8.65]
#     return np.array(array)
#
# def loadGUV1():
#     # Import sampleGUV1 - the 3-dimensional array that holds the fluorescence density for each voxel.'''
#     # With GUV1, 15x15x15 ranges from 15 - 37
#     with open('samples/GUV1trimmed.txt', 'r') as f: text = f.read()
#     for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
#     array = eval(text)
#     boundingBoxDim = [25.95, 25.95, 25.95]
#     return np.array(array), boundingBoxDim
#
# def loadGUV2():
#     # Import sampleGUV1 - the 3-dimensional array that holds the fluorescence density for each voxel.'''
#     # With GUV1, 15x15x15 ranges from 15 - 37
#     #{"GUV center=", {32.005, 32.005, 32.005}, ", GUV radius=", 30.275, ", membrane thick=", 1.73}
#     #{64.01, 64.01, 64.01}
#     with open('samples/GUV2BTrimmed.txt', 'r') as f: text = f.read()
#     for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
#     array = eval(text)
#     boundingBoxDim = [64.01, 64.01, 64.01]
#     return np.array(array), boundingBoxDim

def sample_2by2():
    array = [
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

    return np.array(array)

def sample_1by1():
    array = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    return np.array(array)

def sample_lineY(size):
    array = np.zeros((size, size, size))
    for y in range(array.shape[1]):
        array[int(size/2)][y][int(size/2)] = 1
    return array

def sample_lineZ(size):
    array = np.zeros((size, size, size))
    for z in range(array.shape[2]):
        array[int(size/2)][int(size/2)][z] = 1
    return array

def sample_lineX(size):
    array = np.zeros((size, size, size))
    for x in range(array.shape[0]):
        array[x][int(size/2)][int(size/2)] = 1
    return array


def sample_lineX2(size):
    array = np.zeros((size, size+1, size+1))
    for x in range(array.shape[0]):
        array[x][int(size / 2)][int(size / 2)] = 1
        array[x][int(size / 2) + 1][int(size / 2)] = 1
        array[x][int(size / 2)][int(size / 2) + 1] = 1
        array[x][int(size / 2) + 1][int(size / 2) + 1] = 1
    return array

def sample_diag(size):
    array = np.zeros((size, size, size))
    np.fill_diagonal(array, 1)
    return array

def sample_block(size):
    array = np.ones((size, size, size))
    return array


def showSampleCrossSection(sampleArray):
    # X dim midpoint
    plt.figure("Sample")
    plt.imshow(sampleArray[int(sampleArray.shape[0]/2)], cmap=plt.cm.hot)
    plt.show()

""" Test Samples
Create a 1 voxel sample... at 7,7,7 in 15x15x15 3d array 
shape = [15,15,15]
sampleArray = np.zeros(shape)
sampleArray[8][8][8] = 1.
"""

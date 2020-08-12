import copy
# import numba
#from sparse import COO

from utils import sizeOf
from camRayEntrance import camRayEntrance
import matplotlib.pyplot as plt
import numpy as np
import time
import tifffile
import math

magnObj = 60  # magnification of objective lens
naObj = 1.2  # naObj is NA of objective lens
nMedium = 1.33  # Medium is the refractive index of the medium in object space
nrCamPix = 16  # 16 x 16 pixels
    # nrCamPix is the number of camera pixels behind a lenslet in the horizontal and vertical direction
    # ≤ i ≤ nrCamPix and 0 ≤ j ≤ nrCamPix span the plane of camera pixels, with {i,j} Reals
    # Integers {i,j} count the pixels. {0.5,0.5} is the center location of pixel {1,1}
camPixPitch = 6.5  # size of camera pixels in micron
uLensCtr = {8, 8}  # microLens center position in camera pixels
# uLensPitch = nrCamPix * camPixPitch / magnObj  # µLens pitch in object space
voxPitch = 1.73  # in µm
voxPitchOver1 = 1 / 1.73
extentOfSpaceX = 250  # microns
extentOfSpaceYZ = 700  # microns

def getVoxelDims():
    """ Voxel dimensions, extent...
    voxPitch is the side length in micron of a cubic voxel in object space
    and of a square cell of the entrance and exit face """
    dx = voxPitch
    dy = voxPitch
    dz = voxPitch
    def is_odd(a):
        return bool(a - ((a >> 1) << 1))
    '''voxNrX is the number of voxels along the x-axis side of the object cube. An
    odd number will put the Center of a voxel in the center of object space
    if even, add 1'''
    voxNrX = round(extentOfSpaceX / voxPitch)
    if not is_odd(voxNrX):
        voxNrX = voxNrX + 1
    ''' voxNrYZ is the number of voxels along the y- and z-axis side of the object cube
    An odd number will put the Center of a voxel in the center of object space'''
    voxNrYZ = round(extentOfSpaceYZ / voxPitch)
    if not is_odd(voxNrYZ):
        voxNrYZ = voxNrYZ + 1
    print("voxNrX: ", voxNrX, "voxNrYZ: ", voxNrYZ)
    # voxNrX:  145 voxNrYZ:  405
    # center voxel
    ''' voxCtr is the location in object space on which all camera rays converge
    is a coordinate (not an index)'''
    voxCtr = [voxNrX * voxPitch / 2, voxNrYZ * voxPitch / 2, voxNrYZ * voxPitch / 2]
    # voxCtr is a member of each midpoints list
    print("voxCtr: ", voxCtr)  # >>> voxCtr:  [125.425, 350.325, 350.325]
    return voxCtr, voxNrX, voxNrYZ


# Samples for testing... =====================================================================
def loadSphere2():
    # Import sampleGUV1 - the 3-dimensional array that holds the fluorescence density for each voxel.'''
    # With GUV1, 15x15x15 ranges from 15 - 37
    with open('samples/SolidSphere2Trimmed.txt', 'r') as f: text = f.read()
    for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
    array = eval(text)
    boundingBoxDim = [5.1899999999999995, 5.1899999999999995, 5.1899999999999995]
    return np.array(array), boundingBoxDim

def loadSphere1():
    # Import sampleGUV1 - the 3-dimensional array that holds the fluorescence density for each voxel.'''
    # With GUV1, 15x15x15 ranges from 15 - 37
    with open('samples/SolidSphere1Trimmed.txt', 'r') as f: text = f.read()
    for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
    array = eval(text)
    boundingBoxDim = [8.65, 8.65, 8.65]
    return np.array(array), boundingBoxDim

def loadGUV1():
    # Import sampleGUV1 - the 3-dimensional array that holds the fluorescence density for each voxel.'''
    # With GUV1, 15x15x15 ranges from 15 - 37
    with open('samples/GUV1trimmed.txt', 'r') as f: text = f.read()
    for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
    array = eval(text)
    boundingBoxDim = [25.95, 25.95, 25.95]
    return np.array(array), boundingBoxDim

def loadGUV2():
    # Import sampleGUV1 - the 3-dimensional array that holds the fluorescence density for each voxel.'''
    # With GUV1, 15x15x15 ranges from 15 - 37
    #{"GUV center=", {32.005, 32.005, 32.005}, ", GUV radius=", 30.275, ", membrane thick=", 1.73}
    #{64.01, 64.01, 64.01}
    with open('samples/GUV2BTrimmed.txt', 'r') as f: text = f.read()
    for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
    array = eval(text)
    boundingBoxDim = [64.01, 64.01, 64.01]
    return np.array(array), boundingBoxDim

def sample_2by2():
    array = [
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    boundingBoxDim = [5.1899999999999995, 5.1899999999999995, 5.1899999999999995]
    return np.array(array), boundingBoxDim

def sample_1by1():
    array = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    boundingBoxDim = [5.1899999999999995, 5.1899999999999995, 5.1899999999999995]
    return np.array(array), boundingBoxDim

start = [20, 40, 34]
end = [25, 50, 87]
def sample_diagonal(start, end):
    dx = start[0] - end[0]
    dy = start[1] - end[1]
    dz = start[2] - end[2]
    boundingBoxDim = [dx, dy, dz]
    array=[]
    return np.array(array), boundingBoxDim


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

def getBoundingBox(voxCtr, boundingBoxDim, displace):
    # Bounding Box of Sample, displaced
    """displace = {0 voxPitch, 0 voxPitch, 0 voxPitch}; #displacement from voxCtr
    displace is the displacement vector that moves the center of the bounding box
    containing the simulated object away from the center of object space. A displacement
    along the X-axis will be important for simulating objects that are not in the nominal focal plane. """
    displace = [0 * voxPitch, 0 * voxPitch, 0 * voxPitch]

    ''' voxBoxNrs specify the corner positions of the displaced bounding box in terms of indices (not physical length)
      voxBoxNrs = {{Xmin, Xmax}, {Ymin, Ymax}, {Zmin, Zmax}}
      voxBoxNrs = Transpose[{voxCtr + displace - sample[[2]] / 2, voxCtr + displace + sample[[2]] / 2}]
               / voxPitch // Round;'''

    voxBoxX1 = round((voxCtr[0] + displace[0] - boundingBoxDim[0] / 2) / voxPitch)
    voxBoxX2 = round((voxCtr[0] + displace[0] + boundingBoxDim[0] / 2) / voxPitch)
    voxBoxY1 = round((voxCtr[1] + displace[1] - boundingBoxDim[1] / 2) / voxPitch)
    voxBoxY2 = round((voxCtr[1] + displace[1] + boundingBoxDim[1] / 2) / voxPitch)
    voxBoxZ1 = round((voxCtr[2] + displace[2] - boundingBoxDim[2] / 2) / voxPitch)
    voxBoxZ2 = round((voxCtr[2] + displace[2] + boundingBoxDim[2] / 2) / voxPitch)
    voxBox = [[voxBoxX1, voxBoxX2], [voxBoxY1, voxBoxY2], [voxBoxZ1, voxBoxZ2]]
    print("BoundingBox displaced = ", voxBox)
    return voxBox


def calcSphericalAngles(x1, y1, z1, x2, y2, z2, a_list):
    angles = []
    for m in range(1, len(a_list)):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        dz = float(z2 - z1)
        inclination = math.atan(math.sqrt(dx*dx + dy*dy) / dz)
        azimuth = math.atan(dy / dx)
        angles.append((inclination, azimuth))
    return angles

def genRayAngles(entrance_, exit_, alist_):
        anglesList = []
        for i in range(len(entrance_)):
            x1, y1, z1 = entrance_[i][0], entrance_[i][1], entrance_[i][2]
            x2, y2, z2 = exit_[i][0], exit_[i][1], exit_[i][2]
            angles = calcSphericalAngles(x1, y1, z1, x2, y2, z2, alist_)
            anglesList.append(angles)

        return anglesList


def genMidPtsLengthsAngleswithSiddon(entrance_, exit_, voxBox_):
    import siddon2
    midpointsList = []
    lengthsList = []
    anglesList = []
    longestMidPts = 0
    for i in range(len(entrance_)):
        x1, y1, z1 = entrance_[i][0], entrance_[i][1], entrance_[i][2]
        x2, y2, z2 = exit_[i][0], exit_[i][1], exit_[i][2]
        # print("x1, y1, z1, x2, y2, z2:", x1, y1, z1,"    ", x2, y2, z2)
        # print("Input ", x1, y1, z1, x2, y2, z2, dx, dy, dz)
        # print("            ",pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut)
        dx, dy, dz = voxPitch, voxPitch, voxPitch
        args1 = (x1, y1, z1, x2, y2, z2, dx, dy, dz,
                 voxBox_[0][0], voxBox_[0][1], voxBox_[1][0], voxBox_[1][1], voxBox_[2][0], voxBox_[2][1])
        # args1 = (x1, y1, z1, x2, y2, z2, dx, dy, dz,
        #         pix_numXOut, pix_numXIn, pix_numYOut, pix_numYIn, pix_numZOut, pix_numZIn)
        args = (x1, y1, z1, x2, y2, z2)
        # call rayTrace only once, then pass to midPoints and lengths
        alist = siddon2.raytrace(*args1)
        midpoints = siddon2.midpoints(*args, alist)
        lengths = siddon2.intersect_length(*args, alist)
        if len(midpoints) > longestMidPts:
            longestMidPts = len(midpoints)
        midpointsList.append(midpoints)  # physical coordinates
        lengthsList.append(lengths)
        anglesList = genRayAngles(entrance_, exit_, alist)
    print("longestMidPts: ", longestMidPts)
    return midpointsList, lengthsList, anglesList


def showMidPointsAndLengths(camPix, midpointsList, lengthsList):
    """ Display # of midpoints and sum of lengths for cam array 16x16 """
    image = np.zeros((16, 16))
    for i in range(len(camPix)):  # 164 rays
        image[int(camPix[i][0]-1), int(camPix[i][1])-1] = len(midpointsList[i])
    plt.figure("Number of MidPoints")
    plt.imshow(image, cmap=plt.cm.hot)
    plt.show()

    image = np.zeros((16, 16))
    for i in range(len(camPix)):  # 164 rays
        lengthsSum = 0
        for j in range(len(lengthsList[i])):
            lengthsSum = lengthsSum + lengthsList[i][j]
        image[int(camPix[i][0]-1), int(camPix[i][1])-1] = lengthsSum
    plt.figure("Lengths Sum")
    plt.imshow(image, cmap=plt.cm.hot)
    plt.show()


def padOut(array, newSize):
    # newSize = [x,y,z]
    ''' Sample Array Padding =========================================================================
        Pad sample array out to working object space
        ArrayPad[sample[[3]], ...] expands the original array sample[[3]] by padding it with zeros,
        so the number of voxels span the whole objects space. In case of voxPitch = 1.73, those are
        145 voxels in X-direction, and 405 voxels in Y- and Z-direction.
        The padding is done in such a way that the object is also moved in the direction given by the displacement vector.
        smpArray = ArrayPad[ sample[[3]],
            Reverse[Transpose[(Transpose[voxBoxNrs] - {{0, 0, 0}, {voxNrX, voxNrYZ, voxNrYZ}}) {1, -1}]]];'''

    x_start = round((newSize[0] - array.shape[0]) / 2)
    y_start = round((newSize[1] - array.shape[1]) / 2)
    z_start = round((newSize[2] - array.shape[2]) / 2)
    print("Placement", x_start, x_start + array.shape[0], y_start, y_start + array.shape[1],
          z_start, z_start + array.shape[2])
    result = np.zeros(newSize)
    result[x_start:x_start + array.shape[0], y_start:y_start + array.shape[1], z_start:z_start + array.shape[2]] = array
    '''
    plt.figure()
    plt.imshow(result[72], cmap=plt.cm.hot)
    plt.show()
    '''
    return result


def generateOffsets(midpointsList_, ulenses):
    # Z ==================================================================
    # extract the Z components
    midsZ= []
    for n in range(len(midpointsList_)):
        midsZ.append([])
        for m in range(len(midpointsList_[n])):
           midsZ[n].append(midpointsList_[n][m][2])
    # Generate Offset Z List =================
    midsOffZ = [[] for i in range(ulenses)]
    # arr = [[i*j for j in range(5)] for i in range(10)]
    for l in range(ulenses):
        offsetZ = (l - ulenses/2) * voxPitch
        midsOffZ[l] = copy.deepcopy(midsZ)
        for n in range(len(midsOffZ[l])):
            for m in range(len(midsOffZ[l][n])):
                midsOffZ[l][n][m] = math.ceil((midsOffZ[l][n][m] + offsetZ) * voxPitchOver1)
    # Y ==================================================================
    # extract the Y components
    midsY= []
    for n in range(len(midpointsList_)):
        midsY.append([])
        for m in range(len(midpointsList_[n])):
           midsY[n].append(midpointsList_[n][m][1])
    # Generate Offset Y List =================
    midsOffY = [[] for i in range(ulenses)]
    # arr = [[i*j for j in range(5)] for i in range(10)]
    for l in range(ulenses):
        offsetY = (l - ulenses/2) * voxPitch
        midsOffY[l] = copy.deepcopy(midsY)
        for n in range(len(midsOffY[l])):
            for m in range(len(midsOffY[l][n])):
                midsOffY[l][n][m] = math.ceil((midsOffY[l][n][m] + offsetY) * voxPitchOver1)
    # X ==================================================================
    # Generate (not Offset) X List =================
    midsX= []
    for n in range(len(midpointsList_)):
        midsX.append([])
        for m in range(len(midpointsList_[n])):
           midsX[n].append(math.ceil(midpointsList_[n][m][0] * voxPitchOver1))
    return midsX, midsOffY, midsOffZ

# @jit(nopython=True)
def genLenslet(nZ, nY, camPix, midsX, midsOffY, midsOffZ, lengthsList, anglesList, paddedArray, bigImage):
    maxIntensity = 0
    for i in range(len(camPix)):  # iterate over the 164 rays
        # ??? ellim. - 1's ???
        intensity = 0
        for n in range(len(midsX[i])):
            smpValue = paddedArray[midsX[i][n] - 1, midsOffY[i][n] - 1, midsOffZ[i][n] - 1]
            if smpValue > 0:
                length = lengthsList[i][n]
                # index out of range here ....
                #inclination = anglesList[i](n,0)
                #azimuth = anglesList[i](n, 1)
                # when Polarization matters...
                #print("incl azim: ", inclination, azimuth)
                intensity = intensity + smpValue * length
        if intensity > maxIntensity: maxIntensity = intensity
        # !! Write directly to BigImage ?
        # imgXoff = int(nZ * 16 + camPix[i][0] - 1)
        # imgYoff = int(nY * 16 + camPix[i][1] - 1)
        imgXoff = int(nZ * 16 + camPix[i][0] - 1)
        imgYoff = int(nY * 16 + camPix[i][1] - 1)
        bigImage[imgXoff, imgYoff] = intensity

    return  maxIntensity

# @jit(nopython=True)
# @jit
def genLightField(ulenses, camPix, midsX, midsOffY, midsOffZ, lengthsList, anglesList, paddedArray):
    # Generate Light field image array
    maxMaxIntensity = 0
    bigImage = np.zeros((16 * ulenses, 16 * ulenses), dtype='uint16')
    for k in range(-ulenses // 2, ulenses // 2):
        nZ = int(k + ulenses // 2)
        for j in range(-ulenses // 2, ulenses // 2):
            nY = int(j + ulenses // 2)
            maxIntensity = genLenslet(nZ, nY, camPix, midsX, midsOffY[nY], midsOffZ[nZ], lengthsList, anglesList, paddedArray, bigImage)
            if maxIntensity > maxMaxIntensity: maxMaxIntensity = maxIntensity

    return bigImage, maxMaxIntensity


# Main test

def main():
    # print("Numba version:", numba.__version__)
    # number of microlenses
    ulenses = 16
    print("ulenses: ", ulenses)
    # Sample ================================
    print("Sample: GUV2 ")
    array, boundingBoxDim = loadGUV1()
    # Voxel space
    voxCtr, voxNrX, voxNrYZ = getVoxelDims()
    # Full working space
    # array = []
    # boundingBoxDim = [extentOfSpaceX, extentOfSpaceYZ, extentOfSpaceYZ]
    # Small Cubes
    # array, boundingBoxDim = sample_2by2()
    # showSampleCrossSection(array)
    print("BoundingBoxDim: ", boundingBoxDim)
    displace = [0, 0, 0]
    voxBox = getBoundingBox(voxCtr, boundingBoxDim, displace)
    '''Load camRayEntrance '''
    camPix, entrance, exits = camRayEntrance(voxCtr)  # 164 (x,y), (x, y, z) (x, y, z)
    # print("lengths of camPix, entrance, exit: ", len(camPix), len(entrance), len(exits))
    ''' Rays - Generate midpoints and lengths for the 164 rays... These are in micron, physical dimensions '''
    start = time.time()
    midpointsList, lengthsList, angleList = genMidPtsLengthsAngleswithSiddon(entrance, exits, voxBox)
    end = time.time()
    print("Siddon = %s sec" % (end - start))
    start = time.time()
    showMidPointsAndLengths(camPix, midpointsList, lengthsList)
    midsX, midsOffY, midsOffZ = generateOffsets(midpointsList,ulenses)
    print("midsX:", sizeOf.getsize(midsX))
    print("midsOffY:", sizeOf.getsize(midsOffY))
    print("midsOffZ:", sizeOf.getsize(midsOffZ))
    end = time.time()
    print("generateOffsets = %s sec" % (end - start))
    paddedArray = padOut(array, [voxNrX, voxNrYZ, voxNrYZ])
    start = time.time()
    print("lengthsList,angleList: ", len(lengthsList), len(angleList))
    lfImage, maxIntensity = genLightField(ulenses, camPix, midsX, midsOffY, midsOffZ, lengthsList, angleList, paddedArray)
    end = time.time()
    # =============================================================================
    print("generateLightField = %s sec" % (end - start))
    #print("maxIntensity:", maxIntensity)
        # display light field image
    plt.figure("Image ")
    plt.imshow(lfImage, cmap=plt.cm.gray, vmin=0, vmax=maxIntensity)  # unit = 65535/maxIntensity
    plt.show()
    tifffile.imsave('lfImage.tiff', lfImage)


if __name__ == "__main__":
    main()

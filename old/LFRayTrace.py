import sys
import numba
from numba import jit

from camRayEntrance import camRayEntrance
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import time
import tifffile
import math

# magnification of objective lens
magnObj = 60
# naObj is NA of objective lens
naObj = 1.2
# Medium is the refractive index of the medium in object space
nMedium = 1.33
''' nrCamPix is the number of camera pixels behind a lenslet in the horizontal and vertical direction
≤i≤nrCamPix and 0≤j≤nrCamPix span the plane of camera pixels, with {i,j} Reals
Integers {i,j} count the pixels. {0.5,0.5} is the center location of pixel {1,1}'''
nrCamPix = 16  # 16 x 16 pixels
# size of camera pixels in micron
camPixPitch = 6.5
# microLens center position in camera pixels
uLensCtr = {8, 8}
# µLens pitch in object space
# uLensPitch = nrCamPix * camPixPitch / magnObj

voxPitch = 1.73  # in µm
voxPitchOver1 = 1 / 1.73
extentOfSpaceX = 250  # microns
extentOfSpaceYZ = 700  # microns

def getVoxelDims():
    ''' Voxel dimensions, extent ================================================================
    voxPitch is the side length in micron of a cubic voxel in object space
    and of a square cell of the entrance and exit face '''
    dx = voxPitch
    dy = voxPitch
    dz = voxPitch

    def is_odd(a):
        return bool(a - ((a >> 1) << 1))

    '''voxNrX is the number of voxels along the x-axis side of the object cube. An
    odd number will put the Center of a voxel in the center of object space
    if even, add 1 
      If[OddQ[voxNrX = Round[extentOfSpaceX / voxPitch]], , voxNrX = voxNrX + 1];
    '''
    voxNrX = round(extentOfSpaceX / voxPitch)
    if not is_odd(voxNrX):
        voxNrX = voxNrX + 1
    '''(*voxNrYZ is the number of voxels along the y- and z-axis side of the object cube
    An odd number will put the Center of a voxel in the center of object space
      If[OddQ[voxNrYZ = Round[extentOfSpaceYZ / voxPitch]], , voxNrYZ = voxNrYZ + 1]; '''
    voxNrYZ = round(extentOfSpaceYZ / voxPitch)
    if not is_odd(voxNrYZ):
        voxNrYZ = voxNrYZ + 1
    print("voxNrX: ", voxNrX, "voxNrYZ: ", voxNrYZ)
    # >>> voxNrX:  145 voxNrYZ:  405
    # center voxel
    ''' voxCtr is the location in object space on which all camera rays converge
    is a coordinate (not an index)'''
    voxCtr = [voxNrX * voxPitch / 2, voxNrYZ * voxPitch / 2, voxNrYZ * voxPitch / 2]
    # voxCtr is a member of each midpoints list
    print("voxCtr: ", voxCtr)  # >>> voxCtr:  [125.425, 350.325, 350.325]
    return voxCtr, voxNrX, voxNrYZ


# Samples for testing... =====================================================================
def loadGUV1():
    # Import sampleGUV1 - the 3-dimensional array that holds the fluorescence density for each voxel.'''
    # With GUV1, 15x15x15 ranges from 15 - 37
    with open('../samples/GUV1trimmed.txt', 'r') as f: text = f.read()
    for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
    array = eval(text)
    '''
    for i in range(len(array)):
        for j in range(len(array[i])):
            print(array[i][j], end='\n')
        print()
    '''
    boundingBoxDim = [25.95, 25.95, 25.95]
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







def genSiddon(entrance_, exits_, voxBox_):
    import siddon2
    midpointsList = []
    lengthsList = []
    for i in range(len(entrance_)):
        x1, y1, z1 = entrance_[i][0], entrance_[i][1], entrance_[i][2]
        x2, y2, z2 = exits_[i][0], exits_[i][1], exits_[i][2]
        # Reversed these and it works...

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
        midpointsList.append(midpoints)  # physical coordinates
        lengthsList.append(lengths)
    return np.array(midpointsList), np.array(lengthsList)


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
        ???????????????
          smpArray = ArrayPad[ sample[[3]],
            Reverse[Transpose[(Transpose[voxBoxNrs] - {{0, 0, 0}, {voxNrX, voxNrYZ, voxNrYZ}}) {1, -1}]]];'''

    x_start = round((newSize[0] - array.shape[0]) / 2)
    y_start = round((newSize[1] - array.shape[1]) / 2)
    z_start = round((newSize[2] - array.shape[2]) / 2)
    print("Placement", x_start, x_start + array.shape[0], y_start, y_start + array.shape[1], z_start,
          z_start + array.shape[2])
    result = np.zeros(newSize)
    result[x_start:x_start + array.shape[0], y_start:y_start + array.shape[1], z_start:z_start + array.shape[2]] = array
    return result

    """
    plt.figure()
    plt.imshow(paddedArray[70], cmap=plt.cm.hot)
    plt.show()
    #plt.colorbar()
    """



# @jit
def addOffset(x, o):
    return [int(math.ceil((x[0] + o[0])*voxPitchOver1)),
            int(math.ceil((x[1] + o[1])*voxPitchOver1)),
            int(math.ceil((x[2] + o[2])*voxPitchOver1))]

def addOffsetZ(x, o):
    return [x[0],
            x[1],
            int(math.ceil((x[2] + o[2])*voxPitchOver1))]
def addOffsetY(x, o):
    return [x[0],
            int(math.ceil((x[1] + o[1])*voxPitchOver1)),
            x[2]]

# @jit

def genLenslet(offset, camPix, midPointsList__, lengthsList, paddedArray):
    camArray = []
    maxIntensity = 0
    mids = np.array(midPointsList__)
    # print(k, j, i, camPix[i][0], camPix[i][1]) #, midpointsList[i], lengthsList[i], end='\n')
    # mP = Map[Function[Plus[#, {0, jj µLensPitch, kk µLensPitch}], mP0[[ii]]]
    # vectorize... parallel access?
    # mps = midPointsList__[i]
    # receives midsZ, now only offset Y
    for n in range(len(mids[i])):
        mids[i][n] = addOffsetY(mids[i][n], offset)
    # mids = midPointsList.copy()

    for i in range(len(camPix)):  # for each of 164 rays

        """convert the midpoint coordinates into integers, using Ceiling. 
        The integer coordinates identify the object voxels that are traversed by ray ii. 
        Reverse and Transpose put the converted midpoints into the column and row order 
        required to match the convention used for identifying object voxels."""
        # tmp2 = Ceiling[Transpose[Reverse[Transpose[mP]]]/voxPitch]
        # !! multiply by inverse
        # print("midsOffInt shape, len: ", np.shape(midsOffInt), len(midsOffInt))
        # ??? ellim. - 1's ???
        intensity = 0
        for n in range(len(mids[i])):
            smpValue = paddedArray[mids[i][n][0] - 1, mids[i][n][1] - 1, mids[i][n][2] - 1]
            # print(midsOffInt[n][0]-1, midsOffInt[n][1]-1, midsOffInt[n][2]-1, smpValue, end="   ")
            if smpValue > 0:
                length = lengthsList[i][n]
                intensity = intensity + smpValue * length
        if intensity > maxIntensity: maxIntensity = intensity
        # !! Write directly to BigImage ?
        camArray.append([camPix[i][0] - 1, camPix[i][1] - 1, intensity])
    return  camArray, maxIntensity

'''
    def genLenslet(offset, camPix, midpointsList, lengthsList, paddedArray):
        camArray = []
        maxIntensity = 0
        for i in range(len(camPix)):  # 164 rays
            # print(k, j, i, camPix[i][0], camPix[i][1]) #, midpointsList[i], lengthsList[i], end='\n')
            # mP = Map[Function[Plus[#, {0, jj µLensPitch, kk µLensPitch}], mP0[[ii]]]
            midsOff = []
            # vectorize... parallel access?
            for n in range(len(midpointsList[i])):
                midsOff.append(addOffset(midpointsList[i][n], offset))
            """convert the midpoint coordinates into integers, using Ceiling. 
            The integer coordinates identify the object voxels that are traversed by ray ii. 
            Reverse and Transpose put the converted midpoints into the column and row order 
            required to match the convention used for identifying object voxels."""
            # tmp2 = Ceiling[Transpose[Reverse[Transpose[mP]]]/voxPitch]
            # !! multiply by inverse
            midsOffCeil = np.ceil(np.array(midsOff) * voxPitchOver1)
            midsOffInt = np.int_(midsOffCeil)
            # print("midsOffInt shape, len: ", np.shape(midsOffInt), len(midsOffInt))
            # ??? ellim. - 1's ???
            intensity = 0
            for n in range(len(midsOffInt)):
                smpValue = paddedArray[midsOffInt[n][0] - 1, midsOffInt[n][1] - 1, midsOffInt[n][2] - 1]
                # print(midsOffInt[n][0]-1, midsOffInt[n][1]-1, midsOffInt[n][2]-1, smpValue, end="   ")
                if smpValue > 0:
                    length = lengthsList[i][n]
                    intensity = intensity + smpValue * length
            if intensity > maxIntensity: maxIntensity = intensity
            camArray.append([camPix[i][0] - 1, camPix[i][1] - 1, intensity * 10])
        return camArray, maxIntensity
'''
        # print()
        # print([camPix[i][0]-1, camPix[i][1]-1, intensity/100])
        # camArray[[rayEnterCamPix[[ii, 1]], rayEnterCamPix[[ii, 2]]]] =
        #   Plus @@ (Extract[smpArray, tmp2] Reverse[iL[[ii]]])
        #   or Apply[Plus, Extract[smpArray, tmp2] Reverse[iL[[ii]]]]

# @jit(nopython=True)
# @jit
def genLightField(ulenses, camPix, midpointsList_, lengthsList, paddedArray):

    # Generate Light field image array
    maxMaxIntensity = 0
    bigImage = np.zeros((16 * ulenses, 16 * ulenses)) # , dtype='uint16')
    for k in range(-ulenses // 2, ulenses // 2):
        zOffset = k * voxPitch
        imgXoff = int(k + ulenses / 2) * 16
        # !! do zOffset on midpointsList
        # midsZ =
        for j in range(-ulenses // 2, ulenses // 2):
            yOffset = j * voxPitch
            offset = [0, yOffset, zOffset]
            # pass copy of midsZ
            camArray, maxIntensity = genLenslet(offset, camPix, midpointsList_, lengthsList, paddedArray)
            if maxIntensity > maxMaxIntensity: maxMaxIntensity = maxIntensity
            # ?? mapToLfImage(k, j, camArray)
            imgYoff = int(j + ulenses / 2) * 16 # coords of placement of camArray in bigImage
            # print("k, j, xOff, yOff : %3d %3d    %d %d" % (k, j, xOff, yOff))
            for i in range(len(camArray)):  # 164 rays
                bigImage[imgXoff + int(camArray[i][0]),
                         imgYoff + int(camArray[i][1])] = camArray[i][2]
                print(k, j, yOffset, zOffset, imgXoff + int(camArray[i][0]),  imgYoff + int(camArray[i][1]), camArray[i][2])

    return bigImage, maxMaxIntensity


def showSampleCrossSection(sampleArray):
    # X dim midpoint
    plt.figure("Sample")
    plt.imshow(sampleArray[int(sampleArray.shape[0]/2)], cmap=plt.cm.hot)
    plt.show()


# Main test
def main():
    print("Numba version:", numba.__version__)

    ulenses = 8
    # Voxel space
    voxCtr, voxNrX, voxNrYZ = getVoxelDims()
    # Full working space  boundingBoxDim = [extentOfSpaceX, extentOfSpaceYZ, extentOfSpaceYZ]
    # Sample
    # array, boundingBoxDim = loadGUV1()
    array, boundingBoxDim = sample_1by1()
    # showSampleCrossSection(array)
    print("BoundingBoxDim: ", boundingBoxDim)
    #paddedArray = padOut(array, [voxNrX, voxNrYZ, voxNrYZ])
    #print("paddedArray shape: ", paddedArray.shape)
    displace = [0, 0, 0]
    voxBox = getBoundingBox(voxCtr, boundingBoxDim, displace)
    '''Load camRayEntrance '''
    camPix, entrance, exits = camRayEntrance()  # 164 (x,y), (x, y, z) (x, y, z)
    # print("lengths of camPix, entrance, exit: ", len(camPix), len(entrance), len(exits))
    ''' Rays - Generate midpoints and lengths for the 164 rays... These are in micron, physical dimensions '''
    midpointsList, lengthsList = genSiddon(entrance, exits, voxBox)

    showMidPointsAndLengths(camPix, midpointsList, lengthsList)
    exit(0)
    start = time.time()
    lfImage, maxIntensity = genLightField(ulenses, camPix, midpointsList, lengthsList, paddedArray)
    print("maxIntensity:", maxIntensity)
    # display light field image
    plt.figure("Image ")
    plt.imshow(lfImage, cmap=plt.cm.gray, vmin=0, vmax=maxIntensity)  # unit = 65535/maxIntensity
    plt.show()
    end = time.time()
    print("Calculation = %s" % (end - start))

    tifffile.imsave('lfImage.tiff', lfImage)


if __name__ == "__main__":
    main()

import copy
import multiprocessing
from ctypes import Structure, c_ubyte

from multiprocessing.sharedctypes import RawArray
import struct
#from sparse import COO
from past.builtins import xrange

import samples
from camRayEntrance import camRayEntrance
import matplotlib.pyplot as plt
import numpy as np
from utils import timer
import tifffile
import math
import pickle
from pathlib import Path

# Optical System Parameters ==============================================================

magnObj = 60  # magnification of objective lens
naObj = 1.2  # naObj is NA of objective lens
nMedium = 1.33  # Medium is the refractive index of the medium in object space
nrCamPix = 16  # 16 x 16 pixels
    # nrCamPix is the number of camera pixels behind a lenslet in the horizontal and vertical direction
    # ≤ i ≤ nrCamPix and 0 ≤ j ≤ nrCamPix span the plane of camera pixels, with {i,j} Reals
    # Integers {i,j} count the pixels. {0.5,0.5} is the center location of pixel {1,1}
uLensCtr = {8, 8}  # microLens center position in camera pixels
camPixPitch = 6.5  # size of camera pixels in micron
    # Sensor Pixels = 2048 x 2048
    # (2048 * 6.5 um) / 100 um    133.12
# uLensPitch - µLens pitch in object space.  100 um diameter ulens
# uLens pitch = 1.7333.. or (26/15) microns in object space when using 60x objective lens.
# uLensPitch = (16 pix * 6.5 micron/pix=104 micron/ulens) / 60 = 1.73... microns/ulens in obj space
uLensPitch = nrCamPix * camPixPitch / magnObj
print("uLensPitch:", uLensPitch)

# TODO For different optical configurations, we need a camRayEntrance array.
# opticalConfig:  60x, 1.2 NA  and  20x ? NA


# TODO Axes ??? ray positions on pixels xy from bottom left... ???

def getSpaceDims(extentX_, extentYZ_, voxPitch_, displace_):
    """ Voxel dimensions, extent...
    voxPitch is the side length in micron of a cubic voxel in object space
    and of a square cell of the entrance and exit face """
    print("spaceDims (microns): ", extentX_, extentYZ_, displace_)
    def is_odd(a):
        return bool(a - ((a >> 1) << 1))
    '''voxNrX is the number of voxels along the x-axis side of the object cube. An
    odd number will put the Center of a voxel in the center of object space
    if even, add 1'''
    voxNrX = round(extentX_ / voxPitch_)
    if not is_odd(voxNrX):
        voxNrX = voxNrX + 1
    ''' voxNrYZ is the number of voxels along the y- and z-axis side of the object cube
    An odd number will put the Center of a voxel in the center of object space'''
    voxNrYZ = round(extentYZ_ / voxPitch_)
    if not is_odd(voxNrYZ):
        voxNrYZ = voxNrYZ + 1
    # center voxel
    ''' voxCtr is the location in object space on which all camera rays converge
    is a coordinate (not an index)'''
    voxCtr = [voxNrX * voxPitch_ / 2, voxNrYZ * voxPitch_ / 2, voxNrYZ * voxPitch_ / 2]
    # voxCtr is a member of each midpoints list
    """displace = {0 voxPitch, 0 voxPitch, 0 voxPitch}; #displacement from voxCtr
    displace is the displacement vector that moves the center of the bounding box
    containing the simulated object away from the center of object space. A displacement
    along the X-axis will be important for simulating objects that are not in the nominal focal plane. """
    displace_ = [0 * voxPitch, 0 * voxPitch, 0 * voxPitch]
    # voxBoxNrs specify the corner positions of the displaced bounding box in terms of indices (not physical length)
    voxBoxX1 = round((voxCtr[0] + displace[0] - extentX_ / 2) / voxPitch_)
    voxBoxX2 = round((voxCtr[0] + displace[0] + extentX_/ 2) / voxPitch_)
    voxBoxY1 = round((voxCtr[1] + displace[1] - extentYZ_ / 2) / voxPitch_)
    voxBoxY2 = round((voxCtr[1] + displace[1] + extentYZ_ / 2) / voxPitch_)
    voxBoxZ1 = round((voxCtr[2] + displace[2] - extentYZ_ / 2) / voxPitch_)
    voxBoxZ2 = round((voxCtr[2] + displace[2] + extentYZ_ / 2) / voxPitch_)
    #voxBox = [[voxBoxX1, voxBoxX2], [voxBoxY1, voxBoxY2], [voxBoxZ1, voxBoxZ2]]
    # TODO Forced to Zero... ?
    voxBox = [[0, voxBoxX2], [0, voxBoxY2], [0, voxBoxZ2]]

    return voxNrX, voxNrYZ, voxBox

# =======================================================================================
# Generate mid-points and lengths using Siddon algorithm
# =======================================================================================

def genMidPtsLengthswithSiddon(entrance_, exit_, voxBox_):
    import siddon2
    midpointsList = []
    lengthsList = []
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

    print("longestMidPts: ", longestMidPts)
    # 164 camPixX, camPixY,
    return np.array(midpointsList), np.array(lengthsList)


# not used... for saving and restoring ...
def regeneratePickleSiddon(EXBox_):
    midpointsList, lengthsList, anglesList = genMidPtsLengthswithSiddon(entrance, exits, EXBox_)
    savePickle(midpointsList, 'midpointsList')
    savePickle(lengthsList, 'lengthsList')
    savePickle(anglesList, 'anglesList')
    mids_ = restorePickle('midpointsList')
    lens_ = restorePickle('lengthsList')
    angl_ = restorePickle('anglesList')
    print(mids_ == midpointsList)
    print(lens_ == lengthsList)
    print(angl_ == anglesList)

def savePickle(obj, name):
    f = open(name, 'wb')
    pickle.dump(obj, f)
    f.close()

def restorePickle(name):
    f2 = open(name, 'rb')
    obj = pickle.load(f2)
    f2.close()
    return obj

def showMidPointsAndLengths(camPix, midpointsList, lengthsList):
    """ Diagnostic... Display # of midpoints and sum of lengths for cam array 16x16 """
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

# ==================================================================================
# Generate LightFieldVoxelRay Space
# ==================================================================================
def generateYZOffsets(midpointsList_, ulenses, uLensPitch_, voxPitch_):
    voxPitchOver1 = 1 / voxPitch_
    # Xmin = 10000
    # Xmax = 0
    # Ymin = 10000
    # Ymax = 0
    # Zmin = 10000
    # Zmax = 0
    # Z =======================
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
        offsetZ = (l - ulenses/2) * uLensPitch_
        midsOffZ[l] = copy.deepcopy(midsZ)
        for n in range(len(midsOffZ[l])):
            for m in range(len(midsOffZ[l][n])):
                midsOffZ[l][n][m] = math.ceil((midsOffZ[l][n][m] + offsetZ) * voxPitchOver1)
                #if midsOffZ[l][n][m] > Zmax: Zmax = midsOffZ[l][n][m]
                #if midsOffZ[l][n][m] < Zmin: Zmin = midsOffZ[l][n][m]
    # Y ========================
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
        offsetY = (l - ulenses/2) * uLensPitch_
        midsOffY[l] = copy.deepcopy(midsY)
        for n in range(len(midsOffY[l])):
            for m in range(len(midsOffY[l][n])):
                midsOffY[l][n][m] = math.ceil((midsOffY[l][n][m] + offsetY) * voxPitchOver1)
                #if midsOffY[l][n][m] > Ymax: Ymax = midsOffZ[l][n][m]
                #if midsOffY[l][n][m] < Ymin: Ymin = midsOffZ[l][n][m]
    # X =============================
    # Generate (not Offset) X List
    midsX= []
    for n in range(len(midpointsList_)):
        midsX.append([])
        for m in range(len(midpointsList_[n])):
           x = math.ceil(midpointsList_[n][m][0] * voxPitchOver1)
           midsX[n].append(x)
           #midsX[n].append(math.ceil(midpointsList_[n][m][0] * voxPitchOver1))
           #if x > Xmax: Xmax = x
           #if x < Xmin: Xmin = x

    #print("Xmin,Xmax,Ymin,Ymax,Zmin,Zmax:", Xmin, Xmax, Ymin, Ymax, Zmin, Zmax)
    return midsX, midsOffY, midsOffZ

def genLightFieldVoxels(voxNrX_, voxNrYZ_, ex_obj_offsets_, ulenses, camPix, midsX, midsOffY, midsOffZ, lengthsList, anglesList):

    # [ulenses,ulenses,len(camPix)] each containing [length, alt, azim]
    # Array of empty lists
    #voxel = [[[[] for iZ in xrange(nZ)] for iY in xrange(nY)] for iX in xrange(nX)]
    # Using numpy...
    # voxel = np.empty([voxNrX_, voxNrYZ_, voxNrYZ_], dtype='object')
    # voxel[ [x,y,z], has list of rays: [ray(nRay, nZ, nY, len)] ]

    # ??? class Ray(Structure):
    #        _fields_ = [('nRay', c_ubyte), ('nZ', c_ubyte), ('nY', c_ubyte), ('len', c_ubyte)]

    # TODO parallelization... uLenses / numProcs
    def process_for_k(chunk_):
        # sub-process for each k
        for k in chunk_:
            for j in range(ulenses):
                nZ = k
                nY = j
                for nRay in range(len(camPix)):  # iterate over the 164 rays
                    for midpt in range(len(midsX[nRay])):  # number of midpoints
                        x, y, z = int(midsX[nRay][midpt] - 1 - ex_obj_offsets_[0]), \
                                  int(midsOffY[nY][nRay][midpt] - 1 - ex_obj_offsets_[1]), \
                                  int(midsOffZ[nZ][nRay][midpt] - 1 - ex_obj_offsets_[2])
                        if 0 <= x < voxNrX_ and \
                                0 <= y < voxNrYZ_ and \
                                0 <= z < voxNrYZ_:
                            if voxel[x][y][z] is None:
                                # voxel[x][y][z] =  [[nRay, nZ, nY, lengthsList[nRay][midpt]]]
                                voxel[x][y][z] = [struct.pack('BbbB', nRay, nZ, nY, int(lengthsList[nRay][midpt] * 48))]

                            else:
                                packedRay = struct.pack('BbbB', nRay, nZ, nY, int(lengthsList[nRay][midpt] * 48))
                                voxel[x][y][z].append(packedRay)


    # def chunks(l, n):

    #number_of_rays = 0
    numProc = getNumProcs()
    l = list(range(ulenses))
    chunks_of_k = [l[x: x + numProc] for x in xrange(0, len(l), numProc)]
    for chunk in chunks_of_k:
        process_for_k(chunk)

    #print("number_of_rays:", number_of_rays)
    return voxel

def generateLightFieldVoxelRaySpace(voxNrX_, voxNrYZ_, ex_obj_offsets_,  ulenses_, uLensPitch_, voxPitch_, entrance_, exits_, EXBox_):
    # Rays - Generate midpoints and lengths for the 164 rays... These are in micron, physical dimensions
    # Done been pickled...
    # these are independent of number of lenslets
    #timer.startTime("loading midpointsList, lengthsList, anglesList...")
    #midpointsList = restorePickle('midpointsList')
    #lengthsList = restorePickle('lengthsList')
    #anglesList = restorePickle('anglesList')
    #timer.endTime()
    # temp
    timer.startTime()
    midpointsList, lengthsList = genMidPtsLengthswithSiddon(entrance_, exits_, EXBox_)
    #
    anglesList = genRayAngles(entrance_, exits_)
    timer.endTime("Siddon")
    # given uLenses, gen offsets
    timer.startTime()
    # showMidPointsAndLengths(camPix, midpointsList, lengthsList)
    midsX, midsOffY, midsOffZ = generateYZOffsets(midpointsList, ulenses_,uLensPitch_, voxPitch_)
    timer.endTime("GenerateOffsets")
    #print("midsX:", sizeOf.total_size(midsX))
    #print("midsOffY:", sizeOf.total_size(midsOffY))
    #print("midsOffZ:", sizeOf.total_size(midsOffZ))
    # print("lengthsList,angleList: ", len(lengthsList), len(anglesList))
    timer.startTime()
    voxel = genLightFieldVoxels(voxNrX_, voxNrYZ_, ex_obj_offsets_, ulenses_, camPix, midsX, midsOffY, midsOffZ, lengthsList, anglesList)
    timer.endTime("genLightFieldVoxels")
    #sizeOfLFVox(voxel)
    #print("voxel size: ", sizeOf.getsize(voxel))
    return voxel

# ======================================================
# Saving/Loading LFRTVs
# TODO We may also need to specify: voxPitch, ulenses, entranceExitX, entranceExitYZ, objectSpaceX, objectSpaceYZ

def saveLightFieldVoxelRaySpace(filename, voxel):
    # Save to file...
    np.save(filename, voxel)

def loadLightFieldVoxelRaySpace(filename):
    voxel = np.load(filename+".npy" , allow_pickle=True)
    return voxel

# ====================================================================================================
# Angles in x,y,z components, unit vectors, list of 164 sets of them
def calcUnitVectorAngles(x1, y1, z1, x2, y2, z2):
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    dz = float(z2 - z1)
    len = math.sqrt(dx*dx + dy*dy + dz*dz)
    ux = dx/len
    uy = dy / len
    uz = dz / len
    return [ux, uy, uz]

def genRayAngles(entrance_, exit_):
    anglesList = []
    for i in range(len(entrance_)):
        x1, y1, z1 = entrance_[i][0], entrance_[i][1], entrance_[i][2]
        x2, y2, z2 = exit_[i][0], exit_[i][1], exit_[i][2]
        unitVector = calcUnitVectorAngles(x1, y1, z1, x2, y2, z2)
        anglesList.append(unitVector)
    return anglesList

# ==================================================================================
# Generate LightField Projections
# ==================================================================================

# @jit(nopython=True)
# @jit
multiplier = 1000
def genLightFieldImage(ulenses, camPix, anglesList, voxel, sampleArray):
    # Generate Light field image array
    nonzeroSample = sampleArray.nonzero()
    #bigImage = np.zeros((16 * ulenses, 16 * ulenses)) #, dtype='uint16')
    # TODO Tiff file type?
    bigImage = np.zeros((16 * ulenses, 16 * ulenses), dtype='uint16')
    print('    number_of_nonzero_voxels:', len(nonzeroSample[0]))
    for n in range(len(nonzeroSample[0])):
        #print(nonzeroSample[0][n], nonzeroSample[1][n], nonzeroSample[2][n])
        #print("value = ", sampleArray[nonzeroSample[0][n], nonzeroSample[1][n], nonzeroSample[2][n]])
        value = sampleArray[nonzeroSample[0][n], nonzeroSample[1][n], nonzeroSample[2][n]]
        rays = voxel[nonzeroSample[0][n], nonzeroSample[1][n], nonzeroSample[2][n]]
        number_of_rays = 0
        if rays is None:
            pass
            #print("rays = None in voxel: ", [nonzeroSample[0][n], nonzeroSample[1][n], nonzeroSample[2][n]])
        else:
            for ray in range(len(rays)):
                if rays[ray] is not None:
                    #print(rays[ray])
                    # ray = [nRay, nZ, nY, len, alt, azim]
                    unpackedRay = struct.unpack('BbbB', rays[ray])
                    nRay = unpackedRay[0]
                    nZ = unpackedRay[1]
                    nY = unpackedRay[2]
                    length = unpackedRay[3]
                    # nRay = rays[ray][0]
                    # nZ = rays[ray][1]
                    # nY= rays[ray][2]
                    # length = rays[ray][3]
                    # TODO intensity multiplier ???
                    intensity = value * length/48 * multiplier
                    # TODO angles... anglesList[nRay]... unit vectors...
                    # nRay indexes to angles
                    # map ray to pixel in lenslet(nY, nZ)
                    imgXoff = int(nZ * 16 + camPix[nRay][0] - 1)
                    imgYoff = int(nY * 16 + camPix[nRay][1] - 1)
                    # print(nRay, nZ, nY, length, imgXoff, imgYoff, intensity, bigImage[imgXoff, imgYoff])
                    # Add this contribution to the pixel value
                    bigImage[imgXoff, imgYoff] = bigImage[imgXoff, imgYoff] + intensity
                    number_of_rays += 1
    print("    number_of_rays:", number_of_rays)
    return bigImage

# ===============================================================================================
# Multiprocessing version...

def getNumProcs():
    try:
        numProcessors = multiprocessing.cpu_count()
        print('CPU count:', numProcessors)
    except NotImplementedError:   # win32 environment variable NUMBER_OF_PROCESSORS not defined
        print('Cannot detect number of CPUs')
        numProcessors = 1
    return numProcessors


def genLightFieldImageMultiProcess(ulenses, camPix, anglesList, sampleArray):
    # Generate Light field image array
    # TODO move nonzeroSample to global
    nonZeroSamples = sampleArray.nonzero()

    # TODO move bigImage to global
    # bigImage = np.zeros((16 * ulenses, 16 * ulenses)) #, dtype='uint16')
    # TODO Tiff file type?
    bigImage = np.zeros((16 * ulenses, 16 * ulenses), dtype='uint16')
    # TODO Break up into chunks for multiprocessing...
    number_of_nonzero_voxels = np.count_nonzero(sampleArray) # len(nonZeroSamples)
    print('number_of_nonzero_voxels:', number_of_nonzero_voxels)

    def processChunk(chunk):
        for n in range(len(chunk)):
            value = sampleArray[chunk[n][0], chunk[n][1], chunk[n][2]]
            rays = voxel[chunk[n][0], chunk[n][1], chunk[n][2]]
            if rays is None:
                pass
                # print("rays = None in voxel: ", [nonzeroSample[0][n], nonzeroSample[1][n], nonzeroSample[2][n]])
            else:
                for ray in range(len(rays)):
                    if rays[ray] is not None:
                        unpackedRay = struct.unpack('BbbB', rays[ray])
                        nRay = unpackedRay[0]
                        nZ = unpackedRay[1]
                        nY = unpackedRay[2]
                        length = unpackedRay[3]
                        intensity = value * length / 48 * multiplier
                        # TODO angles... anglesList[nRay]... unit vectors... nRay indexes to angles
                        # map ray to pixel in lenslet(nY, nZ)
                        imgXoff = int(nZ * 16 + camPix[nRay][0] - 1)
                        imgYoff = int(nY * 16 + camPix[nRay][1] - 1)
                        # Add this contribution to the pixel value
                        bigImage[imgXoff, imgYoff] = bigImage[imgXoff, imgYoff] + intensity

    nzsample = np.asarray(nonZeroSamples)
    #print(np.shape(nzsample))
    nzs = np.transpose(nzsample)
    # def chunks(l, n):
    #     return [l[x: x + n] for x in xrange(0, len(l), n)]
    # create numProc chunks...
    numProc = getNumProcs()
    chunks = [nzs[i:i + numProc] for i in range(0, len(nzs), numProc)]
    print('# chunks:', len(chunks))
    for chunk in chunks:
        # add to pool
        processChunk(chunk)

    return bigImage


def padOut(array, newSize, offsets):
    print("array.shape, newSize: ", array.shape, newSize)
    # newSize = [x,y,z]
    ''' Sample Array Padding
        Pad sample array out to working object space
        ArrayPad[sample[[3]], ...] expands the original array sample[[3]] by padding it with zeros,
        so the number of voxels span the whole objects space. In case of voxPitch = 1.73, those are
        145 voxels in X-direction, and 405 voxels in Y- and Z-direction.
        The padding is done in such a way that the object is also moved in the direction given by the displacement vector.
        smpArray = ArrayPad[ sample[[3]],
            Reverse[Transpose[(Transpose[voxBoxNrs] - {{0, 0, 0}, {voxNrX, voxNrYZ, voxNrYZ}}) {1, -1}]]];'''
    # TODO Shouldn't be necessary to padout, just change coordinates...
    x_start = round((newSize[0] - array.shape[0]) / 2) + offsets[0]
    y_start = round((newSize[1] - array.shape[1]) / 2) + offsets[1]
    z_start = round((newSize[2] - array.shape[2]) / 2) + offsets[2]
    # TODO ????
    # x_start = math.floor((newSize[0] - array.shape[0]) / 2)
    # y_start = math.floor((newSize[1] - array.shape[1]) / 2)
    # z_start = math.floor((newSize[2] - array.shape[2]) / 2)
    print("    array.shape: ", array.shape, "  Placement", x_start, x_start + array.shape[0], y_start, y_start + array.shape[1],
          z_start, z_start + array.shape[2])
    result = np.zeros(newSize)
    result[x_start:x_start + array.shape[0], y_start:y_start + array.shape[1], z_start:z_start + array.shape[2]] = array
    return result

def loadSample(name):
    with open('samples/' + name + '.txt', 'r') as f: text = f.read()
    for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
    array = eval(text)
    return np.array(array)

def projectSample(name, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path):
    timer.startTime()
    array = loadSample(name)
    if array.shape[0]> voxNrX or array.shape[1] > voxNrYZ or array.shape[2] > voxNrYZ:
        print("* * * Sample [" + name + "] does not fit in object space. Sample shape: " + str(array.shape))
        return
    offsets = [0, 0, 0]
    sampleArray = padOut(array, [voxNrX, voxNrYZ, voxNrYZ], offsets)
    lfImage = genLightFieldImageMultiProcess(ulenses, camPix, anglesList, sampleArray)
    # lfImage = genLightFieldImage(ulenses, camPix, angleList, voxel, sampleArray)
    timer.endTime("genLightFieldImage: " + name)
    if display_plot:
        plt.figure("LFImage:" + name)
        glfImage = np.power(lfImage, gamma)
        plt.imshow(glfImage, origin='lower', cmap=plt.cm.gray)  # , vmin=0, vmax=maxIntensity)  # unit = 65535/maxIntensity
        plt.interactive(False)
        #plt.show(block=True)
        plt.show()
    filename = 'lfimages/' + path + name
    tifffile.imsave(filename + '.plm.tiff', lfImage)
    print('Generated: ' + filename)

    psvImage = generatePerspectiveImages(lfImage)
    tifffile.imsave(filename + '.plm.psv.tiff', psvImage)


def projectArray(array, name, offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path):
    # for test arrays
    if array.shape[0]> voxNrX or array.shape[1] > voxNrYZ or array.shape[2] > voxNrYZ:
        print("* * * Sample [" + name + "] does not fit in object space. Sample shape: " + str(array.shape))
        return
    sampleArray = padOut(array, [voxNrX, voxNrYZ, voxNrYZ], offsets)
    lfImage = genLightFieldImageMultiProcess(ulenses, camPix, anglesList, sampleArray)
    #lfImage = genLightFieldImage(ulenses, camPix, angleList, voxel, sampleArray)
    if display_plot:
        plt.figure("LFImage:" + name)
        glfImage = np.power(lfImage, gamma)
        plt.imshow(glfImage, origin='lower', cmap=plt.cm.gray)  # , vmin=0, vmax=maxIntensity)  # unit = 65535/maxIntensity
        plt.interactive(False)
        #plt.show(block=True)
        plt.show()
    # save lfImage
    filename = 'lfimages/' + path + name + "_" + str(offsets)
    tifffile.imsave(filename + '.plm.tiff', np.flipud(lfImage))
    print('Generated: ' + filename)

    psvImage = generatePerspectiveImages(lfImage)
    tifffile.imsave(filename + '.plm.psv.tiff', psvImage)

# Perspective Images ======================================================================

def generatePerspectiveImages(lfImage_):
    # Generates (uLenses x uLenses) array of (16 x 16) perspective images
    psvImage = np.zeros((16 * ulenses, 16 * ulenses), dtype='uint16')
    # each subimg is ulense square, calc subImages offsets..
    for sx in range(ulenses):
        for sy in range(ulenses):
            for lx in range(16):
                for ly in range(16):
                    # Lf coord
                    lfX = sx * 16 + lx
                    lfY = sy * 16 + ly
                    psX = lx * ulenses + sx
                    psY = ly * ulenses + sy
                    psvImage[psX][psY] = lfImage_[lfX][lfY]
    return psvImage



display_plot = True
gamma = 1.0  # for matplotlib (not tiff file) images:

def runProjection(voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path):
    offsets = [0, 0, 0]
    # projectArray(samples.sample_lineY(32), "Line Y", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # projectArray(samples.sample_lineZ(32), "Line Z", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # projectArray(samples.sample_lineX(32), "Line X", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # projectArray(samples.sample_lineX2(32), "Line X", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # projectArray(samples.sample_diag(32), "Diagonal", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    projectArray(samples.sample_block(32), "Block16", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # projectArray(samples.sample_block(3), "Block3", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # offsets = [0, 20, 0]
    # projectArray(samples.sample_block(3), "Block3 0200", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # offsets = [0, 0, 20]
    # projectArray(samples.sample_block(3), "Block3 0020", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # offsets = [0, 20, -20]
    # projectArray(samples.sample_block(3), "Block3 020-20", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # offsets = [0, -20, -20]
    # projectArray(samples.sample_block(3), "Block3 -020-20", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel,
    #              path)
    # offsets = [0, -20, 20]
    # projectArray(samples.sample_block(3), "Block3-2020", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel,
    #              path)
    # offsets = [0, 20, 20]
    # projectArray(samples.sample_block(3), "Block3 2020", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel,
    #              path)
    # offsets = [10, 0, 0]
    # projectArray(samples.sample_block(3), "Block3 X 1000", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # offsets = [-3, 0, 0]
    # projectArray(samples.sample_block(3), "Block3 X -300", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # projectArray(samples.sample_1by1(), "1x1", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # projectArray(samples.sample_2by2(), "1x1", offsets, voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # #
    # projectSample('GUV1trimmed', voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # projectSample('GUV2BTrimmed', voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # projectSample('SolidSphere1Trimmed', voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    # projectSample('SolidSphere2Trimmed', voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    projectSample('bundle1_0_0Trimmed', voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    projectSample('bundle2_45_45Trimmed', voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)
    projectSample('bundle3_0_90Trimmed', voxNrX, voxNrYZ, ulenses, camPix, angleList, voxel, path)



# ======================================================================================
# Main...
# ======================================================================================
def main():
    pass


# Input Parameters  ===================================================================================

# voxPitch ==========================================
# voxPitch is the side length in microns of a cubic voxel in object space;
# and of a square cell of the entrance and exit face
# determines digital voxel resolution
# Make uLensPitch/voxPitch an odd integer
# possible values for voxPitch: 3, 1, 1/3, 1/5 of 26/15 (uLensPitch)
# voxPitch = (26/15) * 3
# voxPitch = (26/15) * 1
voxPitch = (26/15) / 3
# voxPitch = (26/15) / 5
print("voxPitch: ", voxPitch)
# Number of microlenses ==============================
# Max # of uLenses = 115
ulenses = 8  # number of
print("ulenses:  ",  ulenses)
# What to do...
createVoxels, saveVoxels, readVoxels, doProjections = True, True, False, True
#createVoxels, saveVoxels, readVoxels, doProjections = False, False, True, True


#============================================================================================
# for naming of output files
parameters = str(ulenses) + '_' + "{:3.3f}".format(voxPitch).replace('.', '_')
print('parameters: ', parameters)
#  Images with different parameters are saved to separate directories
path = str(ulenses) + '/' + "{:3.3f}".format(voxPitch).replace('.', '_') + '/'
print('data file path: ', path)
# create directory for outputs with this set of parameters
Path("lfimages/" + path).mkdir(parents=True, exist_ok=True)
# =====================================
displace = [0, 0, 0]
# EntranceExitSpace ===================
# Entrance, Exit planes are 700 x 700, 250 apart (um)
entranceExitX = 250  # microns
entranceExitYZ = 700  # microns
ex_voxNrX, ex_voxNrYZ, ex_voxBox = getSpaceDims(entranceExitX, entranceExitYZ, voxPitch, displace)  # Voxel space
print("EntranceExitSpace: ", ex_voxNrX, ex_voxNrYZ, ex_voxBox)

# Object space =========================
objectSpaceX = 101  # 100 microns
# YZ size a function of the # of uLens
objectSpaceYZ = ulenses * 26/15  # microns
obj_voxNrX, obj_voxNrYZ, obj_voxBox = getSpaceDims(objectSpaceX, objectSpaceYZ, voxPitch, displace)  # Voxel space
print("Object space: ", obj_voxNrX, obj_voxNrYZ, obj_voxBox)

# Coordinate transform --- object space coordinates relative to entrance/exit planes/volume
# ex_obj_offsets = coordTranform(ex_voxBox, obj_voxBox)
# for the calculation of the LFRayVoxelSpace, object space is placed in the center of entranceExitSpace
# Sample is placed in object space coords
offsetX = (ex_voxBox[0][1] - ex_voxBox[0][0]) / 2 - (obj_voxBox[0][1] - obj_voxBox[0][0]) / 2
offsetY = (ex_voxBox[1][1] - ex_voxBox[1][0]) / 2 - (obj_voxBox[1][1] - obj_voxBox[1][0]) / 2
offsetZ = (ex_voxBox[2][1] - ex_voxBox[2][0]) / 2 - (obj_voxBox[2][1] - obj_voxBox[2][0]) / 2
ex_obj_offsets = [int(offsetX), int(offsetY), int(offsetZ)]
print()
print("EntExit-Obj offsets: ", ex_obj_offsets)
print()

# what is maximum accumulated intensity?
#max_length = round(math.sqrt(obj_voxNrX * obj_voxNrX + obj_voxNrYZ * obj_voxNrYZ + obj_voxNrYZ * obj_voxNrYZ))
#print("max_length:", max_length)

# For multiprocessing...======================================================

# LFImage is read/write shared array, ushort
# LFImage = np.zeros((16 * ulenses, 16 * ulenses), dtype='uint16')
#
# # voxels is read-only shared array of rays
# # voxels[x,y,z][nRays]
# multiprocessing.RawArray()
voxel = np.empty([obj_voxNrX, obj_voxNrYZ, obj_voxNrYZ], dtype='object')

#
# camPix = []
# anglesList = multiprocessing.Array('d',100)

if __name__ == "__main__":

    camPix, entrance, exits = camRayEntrance()  # 164 (x,y), (x, y, z) (x, y, z)
    # print("lengths of camPix, entrance, exit: ", len(camPix), len(entrance), len(exits))
    anglesList = genRayAngles(entrance, exits)

    # LightFieldVoxelRaySpace voxel files are in directory lfvox
    lfvox_filename = "lfvox/lfvox_" + parameters
    if createVoxels:
        # Create and save ==============================================
        voxel = generateLightFieldVoxelRaySpace(obj_voxNrX, obj_voxNrYZ, ex_obj_offsets,
                                               ulenses, uLensPitch, voxPitch, entrance, exits, ex_voxBox)
    if saveVoxels:
        # save to disk ============================================
        saveLightFieldVoxelRaySpace(lfvox_filename, voxel)
    if readVoxels:
        # Read from disk ==========================================
        voxel = loadLightFieldVoxelRaySpace(lfvox_filename)
    print ("LFVox file: "  + lfvox_filename)
    del entrance
    del exits

    # need read-access to voxel, camPix, anglesList

    if doProjections:
        print("Image Size: ", 16 * ulenses, 16 * ulenses)
        runProjection(obj_voxNrX, obj_voxNrYZ, ulenses, camPix, anglesList, voxel, path)

    print("All done.")
    #sys.exit(0)

"""
uLensPitch: 1.7333333333333334
voxPitch:  0.5777777777777778
ulenses:   24
parameters:  24_0_578
path:  24/0_578/
spaceDims (microns):  250 700 [0, 0, 0]
EntranceExitSpace:  433 1213 [[0, 433], [0, 1212], [0, 1212]]
spaceDims (microns):  101 41.6 [0, 0, 0]
Object space:  175 73 [[0, 175], [0, 72], [0, 72]]
"""


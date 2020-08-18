import copy
import math
import struct
import matplotlib.pyplot as plt
import numpy as np


import LFRayTraceVoxParams

from LFRayTraceVoxSpace import getVoxelDims, getWorkingDims
# TODO
#from camRayEntrance import camRayEntrance
from utils import timer, sizeOf

# ==================================================================================
# Generate LightFieldVoxelRay Space
# using parameters from LFRayTraceVoxParams
# ==================================================================================

def genMidPtsLengthswithSiddon(entrance_, exit_, voxBox_, voxPitch_):
    # Generate mid-points and lengths using Siddon algorithm
    import siddon2
    midpointsList = []
    lengthsList = []
    longestMidPts = 0
    num_midpts = 0
    # for each of 164 rays...
    for i in range(len(entrance_)):
        x1, y1, z1 = entrance_[i][0], entrance_[i][1], entrance_[i][2]
        x2, y2, z2 = exit_[i][0], exit_[i][1], exit_[i][2]
        dx, dy, dz = voxPitch_, voxPitch_, voxPitch_
        # print("Siddon Input ", x1, y1, z1, x2, y2, z2, dx, dy, dz, voxBox_)
        args1 = (x1, y1, z1, x2, y2, z2, dx, dy, dz,
                 voxBox_[0][0], voxBox_[0][1], voxBox_[1][0], voxBox_[1][1], voxBox_[2][0], voxBox_[2][1])
        args = (x1, y1, z1, x2, y2, z2)
        # call rayTrace only once, then pass to midPoints and lengths
        alist = siddon2.raytrace(*args1)
        midpoints = siddon2.midpoints(*args, alist)
        lengths = siddon2.intersect_length(*args, alist)
        num_midpts = num_midpts + len(midpoints)
        if len(midpoints) > longestMidPts:
            longestMidPts = len(midpoints)
        midpointsList.append(midpoints)  # physical coordinates
        lengthsList.append(lengths)
    print("        longestMidPts: ", longestMidPts)
    print("        num_midpts   : ", num_midpts)
    # 164 camPixX, camPixY,
    # return np.array(midpointsList), np.array(lengthsList)
    return midpointsList, lengthsList


def showMidPointsAndLengths(camPix, midpointsList, lengthsList):
    """ Diagnostic... Display # of midpoints and sum of lengths for cam array 16x16 """
    image = np.zeros((16, 16))
    for i in range(len(camPix)):  # 164 rays
        # print("midPtsListLen", i, len(midpointsList[i]))
        image[int(camPix[i][0]-1), int(camPix[i][1])-1] = len(midpointsList[i])
    plt.figure("Number of MidPoints")
    # plt.interactive(False)
    plt.show(block=False)
    plt.imshow(image, cmap=plt.cm.hot)
    plt.show()

    image = np.zeros((16, 16))
    for i in range(len(camPix)):  # 164 rays
        image[int(camPix[i][0] - 1), int(camPix[i][1]) - 1] = len(lengthsList[i])
    plt.figure("Number of Lengths")
    plt.show(block=False)
    plt.imshow(image, cmap=plt.cm.hot)
    plt.show()

    image = np.zeros((16, 16))
    for i in range(len(camPix)):  # 164 rays
        lengthsSum = 0
        for j in range(len(lengthsList[i])):
            lengthsSum = lengthsSum + lengthsList[i][j]
        image[int(camPix[i][0]-1), int(camPix[i][1])-1] = lengthsSum
    plt.figure("Lengths Sum")
    plt.show(block=False)
    plt.imshow(image, cmap=plt.cm.hot)
    plt.show()


def generateYZOffsets(midpointsList_, ulenses_, uLensPitch_, voxPitch_):
    # pre-calculates y and z components of the shifted midpoints so as to accelerate
    # the calculation in genLightFieldVoxels...
    # Generate offsets in voxels
    voxPitchOver1 = 1.0 / voxPitch_
    Xmin = 10000
    Xmax = 0
    Ymin = 10000
    Ymax = 0
    Zmin = 10000
    Zmax = 0
    # Z =======================
    # extract the Z components
    midsZ= []
    for n in range(len(midpointsList_)):
        midsZ.append([])
        for m in range(len(midpointsList_[n])):
           midsZ[n].append(midpointsList_[n][m][2])
    # Generate Offset Z List =================
    midsOffZ = [[] for i in range(ulenses_)]
    # TODO added 0.5 ???
    for l in range(ulenses_):
        offsetZ = (l + 0.5 - ulenses_ / 2) * uLensPitch_
        midsOffZ[l] = copy.deepcopy(midsZ)
        # print("l, offsetZ", l, offsetZ)
        for n in range(len(midsOffZ[l])):
            for m in range(len(midsOffZ[l][n])):
                z = midsOffZ[l][n][m]
                zOff = math.ceil((z + offsetZ) * voxPitchOver1)
                midsOffZ[l][n][m] = zOff
                if midsOffZ[l][n][m] > Zmax: Zmax = midsOffZ[l][n][m]
                if midsOffZ[l][n][m] < Zmin: Zmin = midsOffZ[l][n][m]
    # Y ========================
    # extract the Y components
    midsY= []
    for n in range(len(midpointsList_)):
        midsY.append([])
        for m in range(len(midpointsList_[n])):
           midsY[n].append(midpointsList_[n][m][1])
    # Generate Offset Y List =================
    midsOffY = [[] for i in range(ulenses_)]
    for l in range(ulenses_):
        offsetY = (l + 0.5 - ulenses_ / 2) * uLensPitch_
        midsOffY[l] = copy.deepcopy(midsY)
        for n in range(len(midsOffY[l])):
            for m in range(len(midsOffY[l][n])):
                midsOffY[l][n][m] = math.ceil((midsOffY[l][n][m] + offsetY) * voxPitchOver1)
                if midsOffY[l][n][m] > Ymax: Ymax = midsOffZ[l][n][m]
                if midsOffY[l][n][m] < Ymin: Ymin = midsOffZ[l][n][m]
    # X =============================
    # Generate (not Offset) X List
    midsX= []
    for n in range(len(midpointsList_)):
        midsX.append([])
        for m in range(len(midpointsList_[n])):
           x = math.ceil(midpointsList_[n][m][0] * voxPitchOver1)
           midsX[n].append(x)
           #midsX[n].append(math.ceil(midpointsList_[n][m][0] * voxPitchOver1))
           if x > Xmax: Xmax = x
           if x < Xmin: Xmin = x
    print("Xmin,Xmax,Ymin,Ymax,Zmin,Zmax:", Xmin, Xmax, Ymin, Ymax, Zmin, Zmax)
    return midsX, midsOffY, midsOffZ


def genLightFieldVoxels(workingBox, ulenses, camPix, midsX, midsOffY, midsOffZ, lengthsList, anglesList):
    # [ulenses,ulenses,len(camPix)] each containing [length, alt, azim]
    # Array of empty lists -> voxel = [[[[] for iZ in xrange(nZ)] for iY in xrange(nY)] for iX in xrange(nX)]
    wbx = workingBox[0][1] - workingBox[0][0]
    wbyz = workingBox[1][1] - workingBox[1][0]

    voxel = np.empty([wbx, wbyz, wbyz], dtype='object')
    # voxel[ [x,y,z], has list of rays passing through it: [ray(nRay, nZ, nY, len)] ]
    # ??? class Ray(Structure):_fields_ = [('nRay', c_ubyte), ('nZ', c_ubyte), ('nY', c_ubyte), ('len', c_ubyte)]
    # TODO parallelization... uLenses / numProcs
    def process_for_k(chunk_):
        # sub-process for each k
        for k in chunk_:
            for j in range(ulenses):
                nZ = k
                nY = j
                for nRay in range(len(camPix)):  # iterate over the 164 rays
                    # print("nRay, # of Mids", nRay, len(midsX[nRay]))
                    for midpt in range(len(midsX[nRay])):  # number of midpoints on this ray
                        # x, y, z = int(midsX[nRay][midpt]), \
                        #           int(midsOffY[nY][nRay][midpt]), \
                        #           int(midsOffZ[nZ][nRay][midpt])
                        #print("type(midsOffY[nZ][nRay][midpt])", type(midsOffY[nZ][nRay][midpt]))
                        #print("type(workingBox[0][0])", type(workingBox[0][0]))
                        # print("     MidPt: ", midpt, ":",
                        #       midsX[nRay][midpt],
                        #       midsOffY[nY][nRay][midpt],
                        #       midsOffZ[nZ][nRay][midpt],
                        #       "    ",
                        #       workingBox[0][0],
                        #       workingBox[1][0],
                        #       workingBox[2][0]
                        #       )
                        # add this ray in EX space to list of rays for the corresponding voxel coord in working space
                        # x, y, z = int(midsX[nRay][midpt] - workingBox[0][0]), \
                        #           int(midsOffY[nY][nRay][midpt] - workingBox[1][0]), \
                        #           int(midsOffZ[nZ][nRay][midpt] - workingBox[2][0])
                        x, y, z = int(midsX[nRay][midpt] - workingBox[0][0]-1), \
                                  int(midsOffY[nY][nRay][midpt] - workingBox[1][0]-1), \
                                  int(midsOffZ[nZ][nRay][midpt] - workingBox[2][0]-1)
                        if 0 <= x < wbx and 0 <= y < wbyz and  0 <= z < wbyz:
                            #print("     nZ, nY, nRay,   x, y, z:  ", nZ, nY, nRay, "   ", x, y, z)
                            packedRay = struct.pack('BBBH', nRay, nZ, nY,
                                                    int(lengthsList[nRay][midpt] * LFRayTraceVoxParams.length_div))
                            if voxel[x][y][z] is None:
                                voxel[x][y][z] = [packedRay]
                            else:
                                voxel[x][y][z].append(packedRay)
                        #else:
                         #   print("  *  nZ, nY, nRay,   x, y, z:  ", nZ, nY, nRay, "   ", x, y, z)
    # def chunks(l, n):
    #number_of_rays = 0
    numProc = LFRayTraceVoxParams.getNumProcs()
    l = list(range(ulenses))
    chunks_of_k = [l[x: x + numProc] for x in range(0, len(l), numProc)]
    for chunk in chunks_of_k:
        process_for_k(chunk)
    #print("number_of_rays:", number_of_rays)
    return voxel

# def generateLightFieldVoxelRaySpace(ulenses_, uLensPitch_, voxPitch_, entrance_, exits_, workingBox_):
    # timer.startTime()
    # # Rays - Generate midpoints and lengths for the 164 rays... These are in micron, physical dimensions
    # midpointsList, lengthsList = genMidPtsLengthswithSiddon(entrance_, exits_, workingBox_, voxPitch_)
    # print("workingBox_:", workingBox_)
    # print("   len(midpointsList)   :",len(midpointsList))
    # print("   max(lengthsList)     :", max(lengthsList))
    # print("   max(max(lengthsList)):", max(max(lengthsList)))
    # anglesList = LFRayTraceVoxParams.genRayAngles(entrance_, exits_)
    # timer.endTime("        Siddon")
    # # diagnostic...
    # showMidPointsAndLengths(camPix, midpointsList, lengthsList)
    # # given uLenses, gen offsets
    # timer.startTime()
    # midsX, midsOffY, midsOffZ = generateYZOffsets(midpointsList, ulenses_, uLensPitch_, voxPitch_)
    # timer.endTime("        generateOffsets")
    # print("    midsX   :", len(midsX))
    # print("    midsOffY:", len(midsOffY))
    # print("    midsOffZ:", len(midsOffZ))
    # # print("lengthsList,angleList: ", len(lengthsList), len(anglesList))
    #
    # timer.startTime()
    # voxel = genLightFieldVoxels(workingBox_, ulenses_, camPix, midsX, midsOffY, midsOffZ, lengthsList, anglesList)
    # timer.endTime("        genLightFieldVoxels")
    #sizeOfLFVox(voxel)
    #print("voxel size: ", sizeOf.getsize(voxel))
    # return voxel

# ======================================================
# Saving/Loading LFRTVs
# TODO We may also need to specify: voxPitch, ulenses, entranceExitX, entranceExitYZ, objectSpaceX, objectSpaceYZ

def saveLightFieldVoxelRaySpace(filename, voxel):
    # Save to file...
    print("Saving voxels to file: " + filename)
    np.save(filename, voxel)

# def loadLightFieldVoxelRaySpace(filename):
#     voxel = np.load(filename+".npy" , allow_pickle=True)
#     return voxel

# DIAGNOSTIC ===============================
def showRaysInVoxels(voxel):
    # diagnostic: shows number of rays in each voxel
    print("voxel.shape:", voxel.shape)
    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                rays = voxel[x][y][z]
                if rays is None:
                    print( x,y,z, " : -----")
                else:
                    print(x, y, z, " : ", len(rays))
                    for ray in range(len(rays)):
                        if rays[ray] is not None:
                            unpackedRay = struct.unpack('BBBH', rays[ray])
                            nRay = unpackedRay[0]
                            nZ = unpackedRay[1]
                            nY = unpackedRay[2]
                            length = unpackedRay[3]
                        #    print(x,y,z,nRay, nZ, nY, length)


def generateLFRTvoxels(ulenses, voxPitch):
    print("Generating lfvox with ulenses: ", ulenses, "  voxPitch: ", voxPitch)
    voxCtr, voxNrX, voxNrYZ = getVoxelDims(LFRayTraceVoxParams.entranceExitX,
                                           LFRayTraceVoxParams.entranceExitYZ, voxPitch)
    print("   EX space specified: (", LFRayTraceVoxParams.entranceExitX, LFRayTraceVoxParams.entranceExitYZ, "microns )")
    print("   EX space, voxCtr:", LFRayTraceVoxParams.formatList(voxCtr),
          "  size: ", voxNrX, voxNrYZ, voxNrYZ)
    # camPix, entrance, exits, angles = camRayCoord(voxCtr)  # 164 (x,y), (x, y, z) (x, y, z)
    # # print("lengths of camPix, entrance, exit: ", len(camPix), len(entrance), len(exits))
    # anglesList = LFRayTraceVoxParams.genRayAngles(entrance, exits) # ????
    angles = LFRayTraceVoxParams.getAngles()
    camPix, rayEntrFace, rayExitFace = LFRayTraceVoxParams.camRayCoord(voxCtr, angles)
    workingBox = getWorkingDims(voxCtr, ulenses, voxPitch)
    print("     Siddon Calcs...")
    # Rays - Generate midpoints and lengths for the 164 rays... These are in micron, physical dimensions
    midpointsList, lengthsList = genMidPtsLengthswithSiddon(rayEntrFace, rayExitFace, workingBox, voxPitch)
    print("     len(midpointsList)   :", len(midpointsList))
    # print("   max(lengthsList)     :", max(lengthsList))
    print("     max(max(lengthsList)), longest length:", max(max(lengthsList)))
    # showMidPointsAndLengths(camPix, midpointsList, lengthsList)
    # given uLenses, gen offsets
    print("    Offsets...")
    midsX, midsOffY, midsOffZ = generateYZOffsets(midpointsList, ulenses, LFRayTraceVoxParams.uLensPitch, voxPitch)
    print("    midsX   :", len(midsX))
    print("    midsOffY:", len(midsOffY))
    print("    midsOffZ:", len(midsOffZ))
    # print("lengthsList,angleList: ", len(lengthsList), len(anglesList))
    timer.startTime()
    voxel = genLightFieldVoxels(workingBox, ulenses, camPix,
                                midsX, midsOffY, midsOffZ,
                                lengthsList,
                                angles)
    timer.endTime("        genLightFieldVoxels")
    # save to disk ============================================
    parameters, imagepath, lfvoxpath = LFRayTraceVoxParams.file_strings(ulenses, voxPitch)
    saveLightFieldVoxelRaySpace(lfvoxpath + "lfvox_" + parameters, voxel)
    # LightFieldVoxelRaySpace voxel files are saved in the directory corresponding to its parameters
    print('    Saved LightFieldVoxelRaySpace to: ', parameters)
    # showRaysInVoxels(voxel) # diagnostic
    del voxel

# ======================================================================================
def main():
    pass

# globals

voxel = None
# midsX, midsOffY, midsOffZ

if __name__ == "__main__":
    # import sys
    # sys.stdout = open('outputGen.txt', 'wt')
    for ulenses in LFRayTraceVoxParams.ulenseses:
        for voxPitch in LFRayTraceVoxParams.voxPitches:
            generateLFRTvoxels(ulenses, voxPitch)

    print("All done Generating LFRTVoxels.")

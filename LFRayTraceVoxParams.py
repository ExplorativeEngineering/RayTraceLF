import multiprocessing
import math
from pathlib import Path
import numpy as np
import psutil

# ==== INPUTS ==================================================
# voxPitch is the side length in microns of a cubic voxel in object space;
# and of a square cell of the entrance and exit face, it determines digital voxel resolution
# Make uLensPitch/voxPitch an odd integer
# possible values for voxPitch: 3, 1, 1/3, 1/5 of 26/15 (uLensPitch)

#voxPitches = [(26 / 15) * 3,  (26 / 15) * 1,  (26 / 15) / 3,   (26 / 15) / 5]
# or [5.2, 1.73, 0.57, 0.346] microns per voxel
voxPitches = [(26 / 15)]
# ulenseses a list of # of uLenses
# ulenseses = [9, 15, 33, 65, 115]
ulenseses = [65]

# ===================================
displace = [0, 0, 0]
# Entrance, Exit planes are 700 x 700, 250 apart (um)
entranceExitX = 250  # microns
entranceExitYZ = 700  # microns
# Working Space
workingSpaceX = 100  # 100 microns
# workingSpaceYZ is a function of the number of uLenses

# Optical System Parameters ==============================================================
magnObj = 60        # magnification of objective lens
nMedium = 1.33      # refractive index of object medium
naObj = 1.2         # NA of objective lens; for naObj=1.2 (water imm. objective), the tilt angle of ray passing
                    # through edge of aperture is arcSin(1.2/1.33)=64°
rNA = 7.7           # radius of NA in aperture plane behind microlens in fraction of camera pixels;
                    # rNA=7.7 is the measured aperture disc radius for the water imm.
                    # objective lens, a 100µm uLens diameter and 6.5 µm camPix pitch (Orca Flash4)
nrCamPix = 16       # nrCamPix is the number of camera pixels behind a lenslet in the horizontal and vertical direction
                    # 0<=i<=nrCamPix and 0<=j<=nrCamPix span the plane of camera pixels, with i,j Reals
                    # integers [i,j] are the pixel count, with i and j starting at 0 and ending at nrCampix-1;
                    # [0.5,0.5] is the center location of the first pixel [0,0]
uLensCtr = [8.,8.]  # the µLens center is at the pixel border between 8th and 9th pixel, horizontally and vertically

camPixPitch = 6.5   # size of camera pixels in micron
                    # Sensor Pixels = 2048 x 2048 ... (2048 * 6.5 um) / 100 um    133.12

# uLensPitch - µLens pitch in object space.  100 um diameter ulens.
#   uLens pitch = 1.7333.. or (26/15) microns in object space when using 60x objective lens.
#   uLensPitch = (16 pix * 6.5 micron= pix=104 micron/ulens) / 60 = 1.73... microns/ulens in obj space
uLensPitch = nrCamPix * camPixPitch / magnObj

# camPixRays ===============================================================================================
# camPixRays generates a square list that holds values (in radian) for azimuth and tilt angles
# in object space for rays originating in camera pixels [i,j] behind a single lenslet
# The computation implements the sine condition for points in the back focal plane of the objective lens.
# 188 rays

def camPixRays(nrCamPix, uLensCtr, nMedium, naObj, rNA):
    # create a square list with nrCamPix * nrCamPix elements, each set to 0
    angles = [[0.] * nrCamPix for i in range(nrCamPix)]
    for i in range(nrCamPix):
        for j in range(nrCamPix):
            tmp = np.sqrt((i + 0.5 - uLensCtr[0]) ** 2 + (j + 0.5 - uLensCtr[1]) ** 2)
            if tmp <= rNA:
                angles[j][i] = [np.arctan2(i + 0.5 - uLensCtr[0], j + 0.5 - uLensCtr[1]),
                                np.arcsin(tmp / rNA * naObj / nMedium)]
    return angles

def camRayCoord(voxCtr,angles):
    camPix = []
    rayEntrFace = []
    rayExitFace = []
    for i in range(len(angles)):
        for j in range(len(angles[i])):
            if angles[i][j]!=0 :
                camPix.append([i,j])
                tmp=[voxCtr[0]*np.tan(angles[i][j][1])*np.sin(angles[i][j][0]),
                     voxCtr[0]*np.tan(angles[i][j][1])*np.cos(angles[i][j][0])]
                rayEntrFace.append([0,voxCtr[1]+tmp[0],voxCtr[2]+tmp[1]])
                rayExitFace.append([2*voxCtr[0],voxCtr[1]-tmp[0],voxCtr[2]-tmp[1]])
    return camPix,rayEntrFace,rayExitFace

def getAngles():
    angles=camPixRays(nrCamPix,uLensCtr,nMedium,naObj,rNA)
    return angles

# ====================================================================================================
# TODO # Angles in x,y,z components, unit vectors, list of 188 sets of them
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

# ====================================================================================================
# Data Types, Ranges (for encoding)
# TODO what is maximum accumulated intensity?  -- LFImage is now double...
# depends on output image depth
# data type of the resulting LF Image
intensity_multiplier = 1000
length_div = 6000
# lengths as high as 7.9....
# TODO div. length by voxPitch, then sqrt(3) into 64000
#max_length = round(math.sqrt(obj_voxNrX * obj_voxNrX + obj_voxNrYZ * obj_voxNrYZ + obj_voxNrYZ * obj_voxNrYZ))
#print("max_length:", max_length)

#============================================================================================
# For naming directories and files...
# parameters, imagepath, lfvoxpath = file_strings(ulenses, voxPitch)
imagedir = "lfimages"
lfvoxdir = "lfrtvox"

def file_strings(ulenses, voxPitch):
    # for naming of output files
    parameters = str(ulenses) + '_' + "{:3.3f}".format(voxPitch).replace('.', '_')
    # print('parameters: ', parameters)
    #  Images with different parameters are saved to separate directories
    imagepath = imagedir + "/" + str(ulenses) + '/' + "{:3.3f}".format(voxPitch).replace('.', '_') + '/'
    lfvoxpath = lfvoxdir + "/" + str(ulenses) + '/' + "{:3.3f}".format(voxPitch).replace('.', '_') + '/'
    # print('data file path: ', path)
    # create directory for outputs with this set of parameters
    Path(imagepath).mkdir(parents=True, exist_ok=True)
    Path(lfvoxpath).mkdir(parents=True, exist_ok=True)
    #lfvox_filename = "lfvox/lfvox_" + parameters
    return parameters, imagepath, lfvoxpath

# ===============================================================================================
# UTILS

def formatList(l):
    return "[" + ", ".join(["%.3f" % x for x in l]) + "]"


def getNumProcs():
    try:
        numProcessors = multiprocessing.cpu_count()
        print('CPU count:', numProcessors)
    except NotImplementedError:   # win32 environment variable NUMBER_OF_PROCESSORS not defined
        print('Cannot detect number of CPUs')
        numProcessors = 1
    return numProcessors


def getNumCores():
    try:
        numCores = psutil.cpu_count(logical=False)
        print('Core count:', numCores)
    except NotImplementedError:   # win32 environment variable NUMBER_OF_PROCESSORS not defined
        print('Cannot detect number of Cores')
        numCores = 1
    return numCores

if __name__ == "__main__":
    print("Nothing to do... "
          "Run generator or projector.")
    getNumProcs()
    getNumCores()
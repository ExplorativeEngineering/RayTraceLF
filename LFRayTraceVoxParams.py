import multiprocessing
import math
from pathlib import Path
# ==== INPUTS ==================================================
# voxPitch is the side length in microns of a cubic voxel in object space;
# and of a square cell of the entrance and exit face, it determines digital voxel resolution
# Make uLensPitch/voxPitch an odd integer
# possible values for voxPitch: 3, 1, 1/3, 1/5 of 26/15 (uLensPitch)
#voxPitches = [(26 / 15) * 3,  (26 / 15) * 1,  (26 / 15) / 3,   (26 / 15) / 5]
voxPitches = [(26 / 15)]
# ulenseses = [8, 16, 32, 64, 115]
# ulenseses = [9, 25, 49, 81, 115]
ulenseses = [3, 4, 5]


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
    # Sensor Pixels = 2048 x 2048 ... (2048 * 6.5 um) / 100 um    133.12
# uLensPitch - µLens pitch in object space.  100 um diameter ulens
# uLens pitch = 1.7333.. or (26/15) microns in object space when using 60x objective lens.
# uLensPitch = (16 pix * 6.5 micron= pix=104 micron/ulens) / 60 = 1.73... microns/ulens in obj space
uLensPitch = nrCamPix * camPixPitch / magnObj
print("uLensPitch:", uLensPitch)
# TODO For different optical configurations, we need a camRayEntrance array for each.
# opticalConfig:  60x, 1.2 NA  and  20x ? NA


# ================================
# Data Types, Ranges (for encoding)
# TODO what is maximum accumulated intensity?
# depends on output image depth
intensity_multiplier = 1000
length_div = 6000
# lengths as high as 7.9....
# TODO div. length by voxPitch, then sqrt(3) into 64000
#max_length = round(math.sqrt(obj_voxNrX * obj_voxNrX + obj_voxNrYZ * obj_voxNrYZ + obj_voxNrYZ * obj_voxNrYZ))
#print("max_length:", max_length)

# For multiprocessing...======================================================
# LFImage is read/write shared array, ushort
# LFImage = np.zeros((16 * ulenses, 16 * ulenses), dtype='uint16')
#
# # voxels is read-only shared array of rays
# # voxels[x,y,z][nRays]
# multiprocessing.RawArray()
# voxel = np.empty([obj_voxNrX, obj_voxNrYZ, obj_voxNrYZ], dtype='object')
#
# camPix = []
# anglesList = multiprocessing.Array('d',100)


#============================================================================================
#parameters, path, lfvox_filename = file_strings(ulenses, voxPitch)
def file_strings(ulenses, voxPitch):
    # for naming of output files
    parameters = str(ulenses) + '_' + "{:3.3f}".format(voxPitch).replace('.', '_')
    print('parameters: ', parameters)
    #  Images with different parameters are saved to separate directories
    path = "lfimages/" + str(ulenses) + '/' + "{:3.3f}".format(voxPitch).replace('.', '_') + '/'
    print('data file path: ', path)
    # create directory for outputs with this set of parameters
    Path(path).mkdir(parents=True, exist_ok=True)
    lfvox_filename = "lfvox/lfvox_" + parameters
    return parameters, path, lfvox_filename

def formatList(l):
    return "["+", ".join(["%.3f" % x for x in l])+"]"

# ===============================================================================================
# UTILS

def getNumProcs():
    try:
        numProcessors = multiprocessing.cpu_count()
        # print('CPU count:', numProcessors)
    except NotImplementedError:   # win32 environment variable NUMBER_OF_PROCESSORS not defined
        print('Cannot detect number of CPUs')
        numProcessors = 1
    return numProcessors

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

if __name__ == "__main__":
    print("Nothing to do... "
          "Run generator or projector.")
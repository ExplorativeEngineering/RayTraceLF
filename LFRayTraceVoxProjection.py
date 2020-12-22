import ctypes
import multiprocessing
import struct
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import LFRayTraceVoxParams
from LFRayTraceVoxSpace import getVoxelDims, getWorkingDims
from utils import timer, sizeOf
import os
import samples
from numba import jit

# Options -----------
sampledir = "samples"
display_plot = False
gamma = 1.0  # for display of matplotlib (not tiff file) images

# Globals for multiprocessing...======================================================
# lfrtVoxels is read-only shared array of rays, loaded from file
# lfrtVoxels[x,y,z][nRays]
lfrtVoxels = None
#
ulenses = None
voxPitch = None
workingBox = []

# camPix: read-only
camPix = None
# angles: read-only
angles = None

# For a sample:
sampleArray = None
# nonZeroSamples: is read-only shared array of sample object voxels
nonZeroSamples = None
nonZeroSamplesArrayTrans = None
#chunks = None
#
# LFImage, the resulting light field image, is read/write shared array
LFImage = None  # float64
LFImage16 = None # unit16

# For multiprocessing testing...
numProcsList = [1, 2, 4, 12, 16, 24]

# totalRays = 0


# ======================================================
# Loading LFRT voxel model from file
# TODO We may also need to specify: voxPitch, ulenses, entranceExitX, entranceExitYZ, objectSpaceX, objectSpaceYZ

def loadLightFieldVoxelRaySpace(filename):
    try:
        voxel_ = np.load(filename+".npy" , allow_pickle=True)
    except IOError:
        print("Failed to load LightFieldVoxelRaySpace: " + filename)
        return None
    else:
        return voxel_

# ==================================================================================
# Generate LightField Projections

# DIAGNOSTIC
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

#@jit(nopython=True)
# def genLightFieldImage():
#     # without multiprocessing
#     # TODO not used
#     global ulenses
#     global camPix
#     global angles
#     global sampleArray
#     global nonZeroSamples
#
#     # Generate Light field image array
#     #nonzeroSample = sampleArray.nonzero()
#     # TODO Tiff file type?
#     bigImage = np.zeros((16 * ulenses, 16 * ulenses), dtype='uint16')
#     print('    number_of_nonzero_voxels:', len(nonZeroSamples[0]))
#     for n in range(len(nonZeroSamples[0])):
#         #print(nonzeroSample[0][n], nonzeroSample[1][n], nonzeroSample[2][n])
#         #print("value = ", sampleArray[nonzeroSample[0][n], nonzeroSample[1][n], nonzeroSample[2][n]])
#         value = sampleArray[nonZeroSamples[0][n], nonZeroSamples[1][n], nonZeroSamples[2][n]]
#         rays = lfrtVoxels[nonZeroSamples[0][n], nonZeroSamples[1][n], nonZeroSamples[2][n]]
#         number_of_rays = 0
#         if rays is None:
#             pass
#             #print("rays = None in voxel: ", [nonzeroSample[0][n], nonzeroSample[1][n], nonzeroSample[2][n]])
#         else:
#             for ray in range(len(rays)):
#                 if rays[ray] is not None:
#                     #print(rays[ray])
#                     # ray = [nRay, nZ, nY, len, alt, azim]
#                     unpackedRay = struct.unpack('BBBH', rays[ray])
#                     nRay = unpackedRay[0]
#                     nZ = unpackedRay[1]
#                     nY = unpackedRay[2]
#                     length = unpackedRay[3]
#                     intensity = value * length / LFRayTraceVoxParams.length_div * \
#                                 LFRayTraceVoxParams.intensity_multiplier
#                     # TODO angles... anglesList[nRay]... unit vectors...
#                     # nRay indexes to angles
#                     # map ray to pixel in lenslet(nY, nZ)
#                     imgXoff = int(nZ * 16 + camPix[nRay][0] - 1)
#                     imgYoff = int(nY * 16 + camPix[nRay][1] - 1)
#                     #print(nRay, nZ, nY, length, imgXoff, imgYoff, intensity, bigImage[imgXoff, imgYoff])
#                     # Add this contribution to the pixel value
#                     bigImage[imgXoff, imgYoff] = bigImage[imgXoff, imgYoff] + intensity
#                     number_of_rays += 1
#     print("       Number_of_rays:", number_of_rays)
#     return bigImage


# ===============================================================================================
# Multiprocessing version...

def init(shared_array_base, ulenses):
    global LFImage
    LFImage = np.ctypeslib.as_array(shared_array_base.get_obj())
    LFImage = LFImage.reshape(16*ulenses, 16*ulenses)


def processChunk(chunk):
    global camPix
    global angles
    global lfrtVoxels
    global nonZeroSamplesArrayTrans
    global LFImage
    global totalRays

    for n in range(len(chunk)):
        value = sampleArray[chunk[n][0], chunk[n][1], chunk[n][2]]
        # TODO for anisotropic we will also access other scalar values for this voxel
        # get list of rays in this voxel
        rays = lfrtVoxels[chunk[n][0], chunk[n][1], chunk[n][2]]
        # print()
        if rays is None:
            pass
            # print("rays = None in voxel: ", [chundk[n][0], chunk[n][1], chunk[n][2]])
        else:
            # totalRays += len(rays)
            for ray in range(len(rays)):
                if rays[ray] is not None:
                    unpackedRay = struct.unpack('BBBH', rays[ray])
                    nRay = unpackedRay[0]
                    nZ = unpackedRay[1]
                    nY = unpackedRay[2]
                    length = unpackedRay[3]
                    # intensity = value * length / LFRayTraceVoxParams.length_div * \
                    #             LFRayTraceVoxParams.intensity_multiplier
                    intensity = value * length
                    # TODO add anisotropic effects based on angles[nRay]  (unit vectors, nRay indexes to angles)
                    # map ray to pixel in lenslet(nY, nZ)
                    imgXoff = int(nZ * 16 + camPix[nRay][0])
                    imgYoff = int(nY * 16 + camPix[nRay][1])
                    # print("imgXoff, imgYoff, nRay, nZ, nY, length, intensity: ",
                    #      imgXoff, imgYoff, nRay, nZ, nY, length, intensity)
                    # Add this contribution to the pixel value in LFImage
                    LFImage[imgXoff, imgYoff] = LFImage[imgXoff, imgYoff] + intensity


def genLightFieldImageMultiProcess(numProc):
    # Generate Light field image array
    global nonZeroSamples
    global nonZeroSamplesArrayTrans
    global ulenses
    global LFImage
    global LFImage16
    global totalRays
    # global chunks
    # Break up into chunks for multiprocessing...
    # number_of_nonzero_voxels = np.count_nonzero(sampleArray)
    number_of_nonzero_voxels = len(nonZeroSamples[0])
    print('        Number_of_nonzero_voxels:', number_of_nonzero_voxels)
    nonZeroSamplesArray = np.asarray(nonZeroSamples)
    # TODO ?? Transpose again ?? only used for chunking
    nonZeroSamplesArrayTrans = np.transpose(nonZeroSamplesArray)
    # create chunks based on number of processors
    # e.g. def chunks(l, n):  return [l[x: x + n] for x in xrange(0, len(l), n)]
    # numProc = LFRayTraceVoxParams.getNumProcs()
    # numProc=12
    chunks = [nonZeroSamplesArrayTrans[i:i + numProc] for i in range(0, len(nonZeroSamplesArrayTrans), numProc)]
    print("          # procs:", numProc, '  # chunks:', len(chunks))
    # Create shared global LFImage

    LFImage_base = multiprocessing.Array(ctypes.c_double, 16 * ulenses * 16 * ulenses)

    # lock = multiprocessing.Lock()
    # totalRays = multiprocessing.Value(ctypes.c_double, 0.0, lock=lock)
    # totalRays = 0

    # Spawn processes... calling processChunk(chunk)
    pool = multiprocessing.Pool(processes=numProc, initializer=init, initargs=(LFImage_base,ulenses,))
    pool.map(processChunk, chunks)

    # print("totalRays:", totalRays)
    # get results, sum number of rays from each process.
    # results = multiprocessing.Pool(number_of_processes).map(createpdf, data)
    # outputs = [result[0] for result in results]
    # pdfoutput = "".join(outputs)

    LFImage = np.ctypeslib.as_array(LFImage_base.get_obj())
    LFImage = LFImage.reshape(16*ulenses, 16*ulenses)
    # Scale float64 image values into LFImage16, uint16
    maxvalue = np.max(LFImage)
    # print("LFImage maxvalue: ", maxvalue)
    maxunit16 = 65536 # for uint16
    scale = maxunit16/maxvalue
    LFImage = LFImage.astype(np.float64) * scale
    LFImage16 = LFImage.astype(np.uint16)
    return

#===========================================================================================
# Single processor version...

@jit(nopython=True)
def genLightFieldImageSingle():
    # Generate Light field image array
    # global nonZeroSamples
    # #global nonZeroSamplesArrayTrans
    # global ulenses
    # global LFImage
    # global LFImage16
    # global totalRays
    #
    # global camPix
    # global angles
    # global lfrtVoxels

    # global chunks
    # Break up into chunks for multiprocessing...
    # number_of_nonzero_voxels = np.count_nonzero(sampleArray)
    number_of_nonzero_voxels = len(nonZeroSamples[0])
    print('        Number_of_nonzero_voxels:', number_of_nonzero_voxels)
    sampleVoxels = np.transpose(np.asarray(nonZeroSamples))

    LFImage = np.zeros([16 * ulenses, 16 * ulenses], dtype=float)

    for n in range(len(sampleVoxels)):
        value = sampleArray[sampleVoxels[n][0], sampleVoxels[n][1], sampleVoxels[n][2]]
        # TODO for anisotropic we will also access other scalar values for this voxel
        # get list of rays in this voxel
        rays = lfrtVoxels[sampleVoxels[n][0], sampleVoxels[n][1], sampleVoxels[n][2]]
        # print()
        if rays is None:
            pass
            # print("rays = None in voxel: ", [chundk[n][0], chunk[n][1], chunk[n][2]])
        else:
            # totalRays += len(rays)
            for ray in range(len(rays)):
                if rays[ray] is not None:
                    unpackedRay = struct.unpack('BBBH', rays[ray])
                    nRay = unpackedRay[0]
                    nZ = unpackedRay[1]
                    nY = unpackedRay[2]
                    length = unpackedRay[3]
                    # intensity = value * length / LFRayTraceVoxParams.length_div * \
                    #             LFRayTraceVoxParams.intensity_multiplier
                    intensity = value * length
                    # TODO add anisotropic effects based on angles[nRay]  (unit vectors, nRay indexes to angles)
                    # map ray to pixel in lenslet(nY, nZ)
                    imgXoff = int(nZ * 16 + camPix[nRay][0])
                    imgYoff = int(nY * 16 + camPix[nRay][1])
                    # print("imgXoff, imgYoff, nRay, nZ, nY, length, intensity: ",
                    #      imgXoff, imgYoff, nRay, nZ, nY, length, intensity)
                    # Add this contribution to the pixel value in LFImage
                    LFImage[imgXoff, imgYoff] = LFImage[imgXoff, imgYoff] + intensity

    # Scale float64 image values into LFImage16, uint16
    maxvalue = np.max(LFImage)
    # print("LFImage maxvalue: ", maxvalue)
    maxunit16 = 65536 # for uint16
    scale = maxunit16/maxvalue
    LFImage = LFImage.astype(np.float64) * scale
    LFImage16 = LFImage.astype(np.uint16)
    return


def placeInWorkingSpace(array, workingDims, offsets):
    # returns array containing
    # newSize = [x,y,z]
    # map the sample array voxels to the working space
    # TODO if sample array > working space...
    # if sample array > working space, bail out, for now
    # shift the center of the sample to the `center of the working space
    # if ulenses is odd, add [0, 1, 1]
    '''Pad sample array out to working space
    ArrayPad[sample[[3]], ...] expands the original array sample[[3]] by padding it with zeros,
    so the number of voxels span the whole objects space. In case of voxPitch = 1.73, those are
    145 voxels in X-direction, and 405 voxels in Y- and Z-direction.
    The padding is done in such a way that the object is also moved in the direction given by the displacement vector.
    smpArray = ArrayPad[ sample[[3]],
        Reverse[Transpose[(Transpose[voxBoxNrs] - {{0, 0, 0}, {voxNrX, voxNrYZ, voxNrYZ}}) {1, -1}]]];
    '''
    # TODO Shouldn't be necessary to padout, just change coordinates...
    def is_odd(a):
        return bool(a - ((a >> 1) << 1))

    x_start = round((workingDims[0] - array.shape[0]) / 2) + offsets[0]
    y_start = round((workingDims[1] - array.shape[1]) / 2) + offsets[1]
    z_start = round((workingDims[2] - array.shape[2]) / 2) + offsets[2]
    print("        Sample array.shape: ",
          array.shape, " workingDims: ",
          workingDims, "  placement: ",
          x_start, x_start + array.shape[0], ',',
          y_start, y_start + array.shape[1], ',',
          z_start, z_start + array.shape[2])
    result = np.zeros(workingDims)
    result[x_start:x_start + array.shape[0], y_start:y_start + array.shape[1], z_start:z_start + array.shape[2]] = array
    return result


def projectArray(array, name, offsets, path):
    global nonZeroSamples
    global workingBox
    global sampleArray
    global LFImage
    # for test arrays
    workingDimX = workingBox[0][1] - workingBox[0][0]
    workingDimYZ = workingBox[1][1] - workingBox[1][0]
    # Test if Sample will fit in Working Space
    if array.shape[0] + offsets[0] > workingDimX or \
            array.shape[1] + offsets[1] > workingDimYZ or \
            array.shape[2] + offsets[2] > workingDimYZ:
        print("    * * * Sample [" + name + "] does not fit in object space. Sample shape: " + str(array.shape))
        return

    print("-----------------------------------------------")
    print("        Projecting: ", name, "     offset:", offsets)
    sampleArray = placeInWorkingSpace(array, [workingDimX, workingDimYZ, workingDimYZ], offsets)
    nonZeroSamples = sampleArray.nonzero()
    numProc = 12
    # for numProc in numProcsList:
    timer.startTime()
    genLightFieldImageMultiProcess(numProc)
    timer.endTime("        genLightFieldImageMultiProcess of" + name)
    #genLightFieldImageSingle()
    #timer.endTime("        genLightFieldImageSingle of" + name)
    print(" ")


    if display_plot:
        plt.figure("LFImage:" + name)
        glfImage = np.power(LFImage, gamma)
        plt.imshow(glfImage, origin='lower',
                   cmap=plt.cm.gray)  # , vmin=0, vmax=maxIntensity)  # unit = 65535/maxIntensity
        # plt.interactive(False)
        # plt.show(block=True)
        plt.show()
    timer.startTime()
    filename = path + name + "_" + str(offsets)
    tifffile.imsave(filename + '.plm.tiff', np.flipud(LFImage16))
    timer.endTime("        save file of" + name)
    # Generate and Save Perspective Images
    timer.startTime()
    psvImage = generatePerspectiveImages(LFImage16, ulenses)
    tifffile.imsave(filename + '.plm.psv.tiff', np.flipud(psvImage))
    timer.endTime("        generatePerspectiveImages of" + name)
    print('        Generated: ' + filename)


# Perspective Images ======================================================================
def generatePerspectiveImages(lfImage_, ulenses_):
    # Generates (uLenses x uLenses) array of (16 x 16) perspective images, ushort
    psvImage = np.zeros((16 * ulenses_, 16 * ulenses_), dtype='uint16')
    # each subimg is ulense square, calc subImages offsets..
    for sx in range(ulenses_):
        for sy in range(ulenses_):
            for lx in range(16):
                for ly in range(16):
                    # Lf coord
                    lfX = sx * 16 + lx
                    lfY = sy * 16 + ly
                    # psv coord
                    psX = lx * ulenses_ + sx
                    psY = ly * ulenses_ + sy
                    psvImage[psX][psY] = lfImage_[lfX][lfY]
    return psvImage


def loadSample(name):
    with open(sampledir + '/' + name, 'r') as f: text = f.read()
    for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
    array = eval(text)
    return np.array(array)

def projectSample(name, offsets, path):
    # Loads sample data as array
    array = loadSample(name+".txt")
    # transpose to correspond to Mathematica coords.
    projectArray(array.transpose(), name, offsets, path)
    # projectArray(array, name, offsets, workingDimX, workingDimYZ, ulenses, camPix, anglesList, path)


def runProjectionsOnAllSamples(workingDimX, workingDimYZ, ulenses, camPix, angleList, path):
    # run on all sample files in sample directory
    offsets = [0, 0, 0]
    for files in os.listdir(sampledir):
        if os.path.isfile(os.path.join(sampledir, files)):
            print(files)
            projectSample(files, offsets, path)


def runProjections(path):
    offsets = [0, 0, 0]
    linelen = 5
    # projectArray(samples.sample_lineY(linelen), "Line Y", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_lineZ(linelen), "Line Z", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_lineX(linelen), "Line X", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_lineX2(32), "Line X", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_diag(32), "Diagonal", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_block(50,100), "Block16", offsets, path) # Occupies most of working space FOR 115 ULENSES....
    # projectArray(samples.sample_block(50,61), "Block-65", offsets, path) # Occupies most of working space FOR 65 ULENSES....
    # projectArray(samples.sample_block(1), "Block1", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_block(3), "Block3", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # testOffset = 3
    # offsets = [0, testOffset, 0]
    # projectArray(samples.sample_block(3), "Block3 0200", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # offsets = [0, 0, testOffset]
    # projectArray(samples.sample_block(3), "Block3 0020", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # offsets = [0, testOffset, -testOffset]
    # projectArray(samples.sample_block(3), "Block3 020-20", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # offsets = [0, -testOffset, -testOffset]
    # projectArray(samples.sample_block(3), "Block3 -020-20", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # offsets = [0, -testOffset, testOffset]
    # projectArray(samples.sample_block(3), "Block3-2020", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # offsets = [0, testOffset, testOffset]
    # projectArray(samples.sample_block(3), "Block3 2020", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # # offset in X...
    # offsets = [testOffset, 0, 0]
    # projectArray(samples.sample_block(3), "Block3 X 1000", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)

    # projectArray(samples.sample_block(3), "Block3 X -300", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_1by1(), "1x1", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_2by2(), "2x2", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # #
    offsets = [0, 0, 0]
    projectSample('GUV1trimmed', offsets, path)
    # projectSample('GUV2BTrimmed', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('GUV2Testtrimmed', offsets, path)
    # projectSample('SolidSphere1Trimmed.txt', offsets, path)
    # projectSample('SolidSphere2Trimmed.txt', offsets, path)
    # projectSample('bundle1_0_0Trimmed',  offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('bundle2_45_45Trimmed',offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('bundle3_0_90Trimmed', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('Bundle1_azim135_incl45', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('Bundle1_azim135_incl0', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('Bundle1_azim90_incl45', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('Bundle1_azim90_incl0', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('Bundle1_azim45_incl45', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('Bundle1_azim45_incl0', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('Bundle1_azim0_incl90', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('Bundle1_azim0_incl45', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('Bundle1_azim0_incl0', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('Bundle2_azim0_incl90', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('Tripod_1B', offsets, path)
    projectSample('CoboidVesicle_', offsets, path)


# ======================================================================================
# Main...
# ======================================================================================


def projectSamples(_ulenses, _voxPitch):

    global voxPitch
    global ulenses
    global workingBox
    global lfrtVoxels
    global camPix
    global angles

    ulenses = ulenses_
    voxPitch = voxPitch_
    print("Running Projections with ulenses: ", ulenses, "  voxPitch: ", voxPitch)
    voxCtr, voxNrX, voxNrYZ = getVoxelDims(LFRayTraceVoxParams.entranceExitX,
                                           LFRayTraceVoxParams.entranceExitYZ, voxPitch)
    print("    EX space specified: (",
          LFRayTraceVoxParams.entranceExitX, LFRayTraceVoxParams.entranceExitYZ, "microns )")
    print("    EX space, voxCtr:", LFRayTraceVoxParams.formatList(voxCtr),
          "  size: ", voxNrX, voxNrYZ, voxNrYZ)

    # camPix, entrance, exits, angles = camRayCoord(voxCtr)  # 164 (x,y), (x, y, z) (x, y, z)
    # # print("lengths of camPix, entrance, exit: ", len(camPix), len(entrance), len(exits))
    # print("    Loading lfvox w/ ulenses, voxPitch: ", ulenses, voxPitch)
    # anglesList = LFRayTraceVoxParams.genRayAngles(entrance, exits)  # ????
    angles = LFRayTraceVoxParams.getAngles()
    camPix, rayEntrFace, rayExitFace = LFRayTraceVoxParams.camRayCoord(voxCtr, angles)
    workingBox = getWorkingDims(voxCtr, ulenses, voxPitch)

    del rayEntrFace
    del rayExitFace

    # get parameters
    parameters, imagepath, lfvoxpath = LFRayTraceVoxParams.file_strings(ulenses, voxPitch)
    print("    Loading " + lfvoxpath + "lfvox_" + parameters)

    # Load lfrtVoxels
    lfrtVoxels = loadLightFieldVoxelRaySpace(lfvoxpath + "lfvox_" + parameters)
    sizeOf.getMemory()
    # Run the projections of the sample objects
    if lfrtVoxels is not None:
        # showRaysInVoxels(voxel)
        print("    Running projections...(image size: ", ulenses * 16, ")")
        runProjections(imagepath)
        # OR
        # runProjectionsOnAllSamples(imagepath)

if __name__ == "__main__":
    # import sys
    # sys.stdout = open('outputProj.txt', 'wt') # redirect print() output to file
    for ulenses_ in LFRayTraceVoxParams.ulenseses:
        for voxPitch_ in LFRayTraceVoxParams.voxPitches:
            projectSamples(ulenses_, voxPitch_)
    print("======================")
    print("Completed Projections.")

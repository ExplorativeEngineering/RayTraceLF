import struct
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import LFRayTraceVoxParams
import samples
from LFRayTraceVoxSpace import getVoxelDims, getWorkingDims
# from camRayEntrance import camRayEntrance
from utils import timer

# ======================================================
# Saving/Loading LFRTVs
# TODO We may also need to specify: voxPitch, ulenses, entranceExitX, entranceExitYZ, objectSpaceX, objectSpaceYZ

def loadLightFieldVoxelRaySpace(filename):
    voxel = np.load(filename+".npy" , allow_pickle=True)
    return voxel

# ==================================================================================
# Generate LightField Projections
# ==================================================================================
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

# @jit(nopython=True)
# @jit
def genLightFieldImage(ulenses, camPix, anglesList, sampleArray):
    # Generate Light field image array
    nonzeroSample = sampleArray.nonzero()
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
                    unpackedRay = struct.unpack('BBBH', rays[ray])
                    nRay = unpackedRay[0]
                    nZ = unpackedRay[1]
                    nY = unpackedRay[2]
                    length = unpackedRay[3]
                    intensity = value * length / LFRayTraceVoxParams.length_div * \
                                LFRayTraceVoxParams.intensity_multiplier
                    # TODO angles... anglesList[nRay]... unit vectors...
                    # nRay indexes to angles
                    # map ray to pixel in lenslet(nY, nZ)
                    imgXoff = int(nZ * 16 + camPix[nRay][0] - 1)
                    imgYoff = int(nY * 16 + camPix[nRay][1] - 1)
                    #print(nRay, nZ, nY, length, imgXoff, imgYoff, intensity, bigImage[imgXoff, imgYoff])
                    # Add this contribution to the pixel value
                    bigImage[imgXoff, imgYoff] = bigImage[imgXoff, imgYoff] + intensity
                    number_of_rays += 1
    print("       Number_of_rays:", number_of_rays)
    return bigImage



# ===============================================================================================
# Multiprocessing version...

def genLightFieldImageMultiProcess(ulenses, camPix, anglesList, sampleArray):
    # Generate Light field image array
    # TODO move nonzeroSample to global
    nonZeroSamples = sampleArray.nonzero()

    # TODO move bigImage to global
    # bigImage = np.zeros((16 * ulenses, 16 * ulenses)) #, dtype='uint16')
    # TODO Tiff file type?
    bigImage = np.zeros((16 * ulenses, 16 * ulenses), dtype='uint32')
    # TODO Break up into chunks for multiprocessing...
    number_of_nonzero_voxels = np.count_nonzero(sampleArray) # len(nonZeroSamples)
    print('        Number_of_nonzero_voxels:', number_of_nonzero_voxels)

    def processChunk(chunk):
        for n in range(len(chunk)):
            value = sampleArray[chunk[n][0], chunk[n][1], chunk[n][2]]
            rays = voxel[chunk[n][0], chunk[n][1], chunk[n][2]]
            # print()
            if rays is None:
                pass
                # print("rays = None in voxel: ", [nonzeroSample[0][n], nonzeroSample[1][n], nonzeroSample[2][n]])
            else:
                for ray in range(len(rays)):
                    if rays[ray] is not None:
                        unpackedRay = struct.unpack('BBBH', rays[ray])
                        nRay = unpackedRay[0]
                        nZ = unpackedRay[1]
                        nY = unpackedRay[2]
                        length = unpackedRay[3]
                        intensity = value * length / LFRayTraceVoxParams.length_div * \
                                    LFRayTraceVoxParams.intensity_multiplier
                        # TODO angles... anglesList[nRay]... unit vectors... nRay indexes to angles
                        # map ray to pixel in lenslet(nY, nZ)
                        imgXoff = int(nZ * 16 + camPix[nRay][0])
                        imgYoff = int(nY * 16 + camPix[nRay][1])
                        # imgXoff = int(nZ * 16 + camPix[nRay][0] - 1)
                        # imgYoff = int(nY * 16 + camPix[nRay][1] - 1)
                        #print("imgXoff, imgYoff, nRay, nZ, nY, length, intensity: ",
                        #      imgXoff, imgYoff, nRay, nZ, nY, length, intensity)
                        # Add this contribution to the pixel value
                        bigImage[imgXoff, imgYoff] = bigImage[imgXoff, imgYoff] + intensity

    nzsample = np.asarray(nonZeroSamples)
    nzs = np.transpose(nzsample)
    # def chunks(l, n):
    #     return [l[x: x + n] for x in xrange(0, len(l), n)]
    # create numProc chunks...
    numProc = LFRayTraceVoxParams.getNumProcs()
    chunks = [nzs[i:i + numProc] for i in range(0, len(nzs), numProc)]
    # print('# chunks:', len(chunks))
    for chunk in chunks:
        # add to pool
        processChunk(chunk)

    return bigImage


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

    # if is_odd(ulenses):
    #     x_start = round((workingDims[0] - array.shape[0]) / 2) + offsets[0]
    #     y_start = round((workingDims[1] - array.shape[1]) / 2) + offsets[1]
    #     z_start = round((workingDims[2] - array.shape[2]) / 2) + offsets[2]
    # else:
    # x_start = ceil((workingDims[0] - array.shape[0]) / 2) + offsets[0]
    # y_start = ceil((workingDims[1] - array.shape[1]) / 2) + offsets[1]
    # z_start = ceil((workingDims[2] - array.shape[2]) / 2) + offsets[2]

    x_start = round((workingDims[0] - array.shape[0]) / 2) + offsets[0]
    y_start = round((workingDims[1] - array.shape[1]) / 2) + offsets[1]
    z_start = round((workingDims[2] - array.shape[2]) / 2) + offsets[2]
    print("         Sample array.shape: ",
          array.shape, " workingDims: ",
          workingDims, "  placement: ",
          x_start, x_start + array.shape[0], ',',
          y_start, y_start + array.shape[1], ',',
          z_start, z_start + array.shape[2])
    result = np.zeros(workingDims)
    result[x_start:x_start + array.shape[0], y_start:y_start + array.shape[1], z_start:z_start + array.shape[2]] = array
    return result


def projectArray(array, name, offsets, workingDimX, workingDimYZ, ulenses, camPix, anglesList, path):
    # for test arrays
    if array.shape[0] + offsets[0] > workingDimX or \
            array.shape[1] + offsets[1] > workingDimYZ or \
            array.shape[2] + offsets[2] > workingDimYZ:
        print("    * * * Sample [" + name + "] does not fit in object space. Sample shape: " + str(array.shape))
        return
    print("        Projecting: ", name, "   ", offsets)
    sampleArray = placeInWorkingSpace(array, [workingDimX, workingDimYZ, workingDimYZ], offsets)
    lfImage = genLightFieldImageMultiProcess(ulenses, camPix, anglesList, sampleArray)
    # lfImage = genLightFieldImage(ulenses, camPix, anglesList, sampleArray)
    if display_plot:
        plt.figure("LFImage:" + name)
        glfImage = np.power(lfImage, gamma)
        plt.imshow(glfImage, origin='lower',
                   cmap=plt.cm.gray)  # , vmin=0, vmax=maxIntensity)  # unit = 65535/maxIntensity
        # plt.interactive(False)
        # plt.show(block=True)
        plt.show()
    filename = path + name + "_" + str(offsets)
    tifffile.imsave(filename + '.plm.tiff', np.flipud(lfImage))
    psvImage = generatePerspectiveImages(lfImage)
    tifffile.imsave(filename + '.plm.psv.tiff', np.flipud(psvImage))
    print('        Generated: ' + filename)


# Perspective Images ======================================================================

def generatePerspectiveImages(lfImage_):
    # Generates (uLenses x uLenses) array of (16 x 16) perspective images, ushort
    psvImage = np.zeros((16 * ulenses, 16 * ulenses), dtype='uint16')
    # each subimg is ulense square, calc subImages offsets..
    for sx in range(ulenses):
        for sy in range(ulenses):
            for lx in range(16):
                for ly in range(16):
                    # Lf coord
                    lfX = sx * 16 + lx
                    lfY = sy * 16 + ly
                    # psv coord
                    psX = lx * ulenses + sx
                    psY = ly * ulenses + sy
                    psvImage[psX][psY] = lfImage_[lfX][lfY]
    return psvImage


sampledir = "samples"

def loadSample(name):
    with open(sampledir + '/' + name + '.txt', 'r') as f: text = f.read()
    for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
    array = eval(text)
    return np.array(array)

def projectSample(name, offsets, workingDimX, workingDimYZ, ulenses, camPix, anglesList, path):
    print("        Projecting: " + name)
    timer.startTime()
    array = loadSample(name)
    projectArray(array.transpose(), name, offsets, workingDimX, workingDimYZ, ulenses, camPix, anglesList, path)
    # projectArray(array, name, offsets, workingDimX, workingDimYZ, ulenses, camPix, anglesList, path)

# Inputs =================================================================================
display_plot = False
gamma = 1.0  # for matplotlib (not tiff file) images:

def runProjections(workingDimX, workingDimYZ, ulenses, camPix, angleList, path):
    offsets = [0, 0, 0]
    linelen = 5
    # projectArray(samples.sample_lineY(linelen), "Line Y", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_lineZ(linelen), "Line Z", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_lineX(linelen), "Line X", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_lineX2(32), "Line X", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_diag(32), "Diagonal", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectArray(samples.sample_block(16), "Block16", offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
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
    projectSample('GUV1trimmed', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    # projectSample('GUV2BTrimmed', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('GUV2Testtrimmed', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('SolidSphere1Trimmed', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('SolidSphere2Trimmed', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('bundle1_0_0Trimmed',  offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('bundle2_45_45Trimmed',offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('bundle3_0_90Trimmed', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('Bundle1_azim135_incl45', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('Bundle1_azim135_incl0', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('Bundle1_azim90_incl45', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('Bundle1_azim90_incl0', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('Bundle1_azim45_incl45', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('Bundle1_azim45_incl0', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('Bundle1_azim0_incl90', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('Bundle1_azim0_incl45', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('Bundle1_azim0_incl0', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)
    projectSample('Bundle2_azim0_incl90', offsets, workingDimX, workingDimYZ, ulenses, camPix, angleList, path)

# ======================================================================================
# Main...
# ======================================================================================

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
# voxel = np.empty([obj_voxNrX, obj_voxNrYZ, obj_voxNrYZ], dtype='object')
# camPix = []
# anglesList = multiprocessing.Array('d',100)

voxel = None

if __name__ == "__main__":
    # import sys
    # sys.stdout = open('outputProj.txt', 'wt') # redirect print() output to file
    for ulenses in LFRayTraceVoxParams.ulenseses:
        for voxPitch in LFRayTraceVoxParams.voxPitches:
            print("Running Projections with ulenses: ", ulenses, "  voxPitch: ", voxPitch)
            voxCtr, voxNrX, voxNrYZ = getVoxelDims(LFRayTraceVoxParams.entranceExitX,
                                                   LFRayTraceVoxParams.entranceExitYZ, voxPitch)
            print("    EX space specified: (", LFRayTraceVoxParams.entranceExitX, LFRayTraceVoxParams.entranceExitYZ,
                  "microns )")
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
            parameters, imagepath, lfvoxpath = LFRayTraceVoxParams.file_strings(ulenses, voxPitch)
            print("    Loading "+ lfvoxpath + "lfvox_" + parameters)
            voxel = loadLightFieldVoxelRaySpace(lfvoxpath + "lfvox_" + parameters)
            # showRaysInVoxels(voxel)
            print("    Running projections...(image size: ", ulenses * 16, ")")
            workingDimX = workingBox[0][1] - workingBox[0][0]
            workingDimYZ = workingBox[1][1] - workingBox[1][0]
            runProjections(workingDimX, workingDimYZ, ulenses, camPix, angles, imagepath)
    print("All done.")

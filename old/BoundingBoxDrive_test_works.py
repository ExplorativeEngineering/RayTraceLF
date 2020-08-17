from oldcamray.camRayEntrance import camRayEntrance
import matplotlib.pyplot as plt
import numpy as np
import time

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
uLensPitch = nrCamPix * camPixPitch / magnObj
print("uLensPitch:", uLensPitch)
''' voxPitch is the side length in micron of a cubic voxel in object space
and of a square cell of the entrance and exit face '''
voxPitch = 1.73  # in µm
dx = voxPitch
dy = voxPitch
dz = voxPitch
extentOfSpaceX = 250  # microns
extentOfSpaceYZ = 700  # microns


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

# With GUV1, 15x15x15 ranges from 15 - 37

''' Sample... ============================================================================================
Import sampleGUV1 - the 3-dimensional array that holds the fluorescence density for each voxel.'''


def loadGUV1():
    with open('../samples/GUV1trimmed.txt', 'r') as f: text = f.read()
    for rep in (('{', '['), ('}', ']')): text = text.replace(rep[0], rep[1])
    array = eval(text)
    '''
    for i in range(len(array)):
        for j in range(len(array[i])):
            print(array[i][j], end='\n')
        print()
    '''
    return np.array(array)


# array = loadGUV1()

""" Test Samples
Create a 1 voxel sample... at 7,7,7 in 15x15x15 3d array 
shape = [15,15,15]
sampleArray = np.zeros(shape)
sampleArray[8][8][8] = 1.
"""
"""
array = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
"""

array = [
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

sampleArray = np.array(array)

"""
plt.figure("Sample[1]")
plt.imshow(sampleArray[1], cmap=plt.cm.hot)
plt.show()
"""

# Bounding Box of Sample, displaced ======================================================================
# boundingBoxDim = [25.95, 25.95, 25.95]
# boundingBoxDim = [5.1899999999999995, 5.1899999999999995, 5.1899999999999995]
boundingBoxDim = [10,10,10]
print("BoundingBoxDim: ", boundingBoxDim)

''' displace = {0 voxPitch, 0 voxPitch, 0 voxPitch}; #displacement from voxCtr
displace is the displacement vector that moves the center of the bounding box 
containing the simulated object away from the center of object space. A displacement 
along the X-axis will be important for simulating objects that are not in the nominal focal plane. '''
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

pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut = \
    voxBoxX1, voxBoxX2, voxBoxY1, voxBoxY2, voxBoxZ1, voxBoxZ2
print("BoundingBox displaced = ", voxBox)
# >>> BoundingBox displaces =  [[65, 80], [195, 210], [195, 210]]

'''Load camRayEntrance ======================================================================='''
camPix, entrance, exits = camRayEntrance(voxCtr)  # 164 (x,y), (x, y, z) (x, y, z)
# print("lengths of camPix, entrance, exit: ", len(camPix), len(entrance), len(exits))

''' Rays - Generate midpoints and lengths for the 164 rays... ===================================
These are in micron, physical dimensions '''
midpointsList = []
lengthsList = []
import siddon2

for i in range(len(entrance)):
    x1, y1, z1 = entrance[i][0], entrance[i][1], entrance[i][2]
    x2, y2, z2 = exits[i][0], exits[i][1], exits[i][2]
    # Reversed these and it works...

    # print("x1, y1, z1, x2, y2, z2:", x1, y1, z1,"    ", x2, y2, z2)
    # print("Input ", x1, y1, z1, x2, y2, z2, dx, dy, dz)
    # print("            ",pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut)
    args1 = (x1, y1, z1, x2, y2, z2, dx, dy, dz,
             pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut)
    # args1 = (x1, y1, z1, x2, y2, z2, dx, dy, dz,
    #         pix_numXOut, pix_numXIn, pix_numYOut, pix_numYIn, pix_numZOut, pix_numZIn)
    args = (x1, y1, z1, x2, y2, z2)
    # call rayTrace only once, then pass to midPoints and lengths
    alist = siddon2.raytrace(*args1)
    midpoints = siddon2.midpoints(*args, alist)
    lengths = siddon2.intersect_length(*args, alist)
    midpointsList.append(midpoints)  # physical coordinates
    lengthsList.append(lengths)

print("camPix length:", len(camPix))

"""
for i in range(len(camPix)):
    print(len(midpointsList[i]), end=',')
    #print("(", camPix[i][0], ',', camPix[i][1], ")   #midpoints/lengths: ", len(midpointsList[i]), end='\n')
print()
"""

""" Display # of midpoints and sum of lengths for cam array 16x16 """
"""
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
"""

''' Sample Array Padding =========================================================================
Pad sample array out to working object space
ArrayPad[sample[[3]], ...] expands the original array sample[[3]] by padding it with zeros, 
so the number of voxels span the whole objects space. In case of voxPitch = 1.73, those are 
145 voxels in X-direction, and 405 voxels in Y- and Z-direction.
The padding is done in such a way that the object is also moved in the direction given by the displacement vector.
???????????????
  smpArray = ArrayPad[ sample[[3]],
    Reverse[Transpose[(Transpose[voxBoxNrs] - {{0, 0, 0}, {voxNrX, voxNrYZ, voxNrYZ}}) {1, -1}]]];'''


def padOut(array, newSize):
    # newSize = [x,y,z]
    x_start = round((newSize[0] - array.shape[0]) / 2)
    y_start = round((newSize[1] - array.shape[1]) / 2)
    z_start = round((newSize[2] - array.shape[2]) / 2)
    print("Placement", x_start, x_start + array.shape[0], y_start, y_start + array.shape[1], z_start,
          z_start + array.shape[2])
    result = np.zeros(newSize)
    result[x_start:x_start + array.shape[0], y_start:y_start + array.shape[1], z_start:z_start + array.shape[2]] = array
    return result


paddedArray = padOut(sampleArray, [voxNrX, voxNrYZ, voxNrYZ])
print("paddedArray shape: ", paddedArray.shape)
"""
plt.figure()
plt.imshow(paddedArray[70], cmap=plt.cm.hot)
plt.show()
#plt.colorbar()
"""

''' Light field image array ======================================================================
'''


# lfImageArray =
def addOffset(x, o):
    return [x[0] + o[0], x[1] + o[1], x[2] + o[2]]


camArray = []
images = []
start = time.time()
maxIntensity = 0

ulenses = 7
bigImage = np.zeros((16 * ulenses, 16 * ulenses), dtype='uint16')

for k in range(-ulenses // 2, ulenses // 2):
    for j in range(-ulenses // 2, ulenses // 2):
        yOffset = j * uLensPitch
        zOffset = k * uLensPitch
        offset = [0, yOffset, zOffset]
        for i in range(len(camPix)):  # 164 rays
            # print(k, j, i, camPix[i][0], camPix[i][1]) #, midpointsList[i], lengthsList[i], end='\n')
            # mP = Map[Function[Plus[#, {0, jj µLensPitch, kk µLensPitch}], mP0[[ii]]]
            midsOff = []
            # imgXOffset = j * ulenses
            # imgYOffset = k * ulenses
            for n in range(len(midpointsList[i])):
                midsOff.append(addOffset(midpointsList[i][n], offset))
            # print(midsOff)
            """convert the midpoint coordinates into integers, using Ceiling. 
            The integer coordinates identify the object voxels that are traversed by ray ii. 
            Reverse and Transpose put the converted midpoints into the column and row order 
            required to match the convention used for identifying object voxels."""
            # tmp2 = Ceiling[Transpose[Reverse[Transpose[mP]]]/voxPitch]
            midsOffCeil = np.ceil(np.array(midsOff) / voxPitch)
            midsOffInt = np.int_(midsOffCeil)
            # print("midsOffInt shape, len: ", np.shape(midsOffInt), len(midsOffInt))
            intensity = 0
            for n in range(len(midsOffInt)):
                smpValue = paddedArray[midsOffInt[n][0] - 1, midsOffInt[n][1] - 1, midsOffInt[n][2] - 1]
                # print(midsOffInt[n][0]-1, midsOffInt[n][1]-1, midsOffInt[n][2]-1, smpValue, end="   ")
                if smpValue > 0:
                    length = lengthsList[i][n]
                    intensity = intensity + smpValue * length
            if (intensity > maxIntensity): maxIntensity = intensity
            camArray.append([camPix[i][0] - 1, camPix[i][1] - 1, intensity * 10])  # div by 100
            # print()
            # print([camPix[i][0]-1, camPix[i][1]-1, intensity/100])
            # camArray[[rayEnterCamPix[[ii, 1]], rayEnterCamPix[[ii, 2]]]] =
            #   Plus @@ (Extract[smpArray, tmp2] Reverse[iL[[ii]]])
            #   or Apply[Plus, Extract[smpArray, tmp2] Reverse[iL[[ii]]]]
        # coords of placement of camArray in bigImage
        nx = int(k + ulenses / 2)
        ny = int(j + ulenses / 2)
        xOff = nx * 16
        yOff = ny * 16
        #print("k, j, nx, ny, xOff, yOff : %3d %3d   %d %d  %d %d" % (k, j, nx, ny, xOff, yOff))
        for i in range(len(camArray)):  # 164 rays
            bigImage[xOff + int(camArray[i][0]), yOff + int(camArray[i][1])] = camArray[i][2]

        ''' map camArray to imageShow lenslet image 16x16 '''
        '''
        image = np.zeros((16, 16), dtype='uint16')
        for i in range(len(camArray)):  # 164 rays
            image[int(camArray[i][0]), int(camArray[i][1])] = camArray[i][2]
        images.append(image)
        '''

plt.figure("Image ")
        #plt.imshow(image, cmap=plt.cm.hot)
plt.imshow(bigImage, cmap=plt.cm.gray, vmin=0, vmax=maxIntensity * 10)
plt.show()

print("maxIntensity = ", maxIntensity)

# unit = 65535/maxIntensity

end = time.time()
print("Calculation = %s" % (end - start))
"""
@jit(nopython=True)
def go_fast(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace
"""

# np.asarray(images, dtype='uint16')
# tifffile.imsave('myimages.tiff', images)
"""
start = time.time()
fig = plt.figure(figsize=(8., 8.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(ulenses, ulenses),  # creates 2x2 grid of axes
                 axes_pad=0.,  # pad between axes in inch.
                 )
end = time.time()
print("ImageGrid = %s" % (end - start))
start = time.time()
for ax, im in zip(grid, images):
    # Iterating over the grid returns the Axes.
    ax.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=1 )
plt.show()
end = time.time()
print("imgshow  = %s" % (end - start))
"""

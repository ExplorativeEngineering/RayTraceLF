from siddon2 import midpoints, intersect_length, raytrace
from camRayEntrance import camRayEntrance
import matplotlib.pyplot as plt
import numpy as np

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
voxPitch = 26/15  # in µm
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

# center voxel
''' voxCtr is the location in object space on which all camera rays converge, it is a coordinate (not an index)'''
voxCtr = [voxNrX * voxPitch/2, voxNrYZ * voxPitch/2, voxNrYZ * voxPitch/2]
# voxCtr is a member of each midpoints list
print("voxCtr: ", voxCtr)  # >>> voxCtr:  [125.6666, 351, 351]

#voxCtr = [125.666666,351,351]

camPix, entrance, exit = camRayEntrance(voxCtr)

print("length of camPix: ",len(camPix), ", length of entrance: ",len(entrance), ", length of exit: ",len(exit))
print("entrance[0]: ",entrance[0])
print("entrance[0]-voxCtr: ",np.subtract(entrance[0],[0,voxCtr[1],voxCtr[2]]))
print("entrance[5]-voxCtr: ",np.subtract(entrance[5],[0,voxCtr[1],voxCtr[2]]))
print("entrance[158]-voxCtr: ",np.subtract(entrance[158],[0,voxCtr[1],voxCtr[2]]))
print("entrance[163]-voxCtr: ",np.subtract(entrance[163],[0,voxCtr[1],voxCtr[2]]))

'''
x1, y1, z1 = 0.0, 281.03366217675773, 281.03366217675773
x2, y2, z2 = 2*voxCtr[0], voxCtr[1]-(y1-voxCtr[1]), voxCtr[2]-(z1-voxCtr[2])
dx, dy, dz = 26/15, 26/15, 26/15
print("ray exit",x2,y2,z2)

#pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut = 65, 80, 195, 210, 195, 210
# 112, 138, 337, 363, 337, 363,

pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut = 72, 73, 202, 203, 202, 203
"""

x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 1, 1, 1
dx, dy, dz = 0.1, 0.1, 0.1
pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut = 0, 10, 0, 10, 0,10
"""


args1 = (x1, y1, z1, x2, y2, z2, dx, dy, dz,pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut)
args = (x1, y1, z1, x2, y2, z2)

# Test Siddon
# call rayTrace only once, then pass to midPoints and lengths
alist = raytrace(*args1)

midpoints1 = midpoints(*args, alist)
lengths = intersect_length(*args, alist)
print('Midpoints: ' + str([(round(x[0], 3), round(x[1], 3), round(x[2], 3)) for x in midpoints1]))
print('Lengths: ' + str([round(x, 3) for x in lengths]))

# Check lengths
from math import sqrt
print("#midpoints: ", len(midpoints1))
print("#lengths: ",len(lengths))
print('Input length: ' + str(sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)))
print('Output length: ' + str(sum(lengths)))
'''
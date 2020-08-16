# 15 Aug 2020, RO
# Definition of methods camPixRays and camRayCoord
# camPixRays generates a square list (nrCamPix*nrCamPix, typically 16*16 elements) that holds values for azimuth and tilt angles in object space for rays originating in camera pixels [i,j] behind a single lenslet
# Pixels near the edge of the array that correspond to rays outside the numerical aperture of the objective lens are set to zero
# The computation implements the sine condition for points in the back focal plane of the objective lens.
# camRayCoord generates an array camPix of pixel indices and two arrays rayEntrFace and rayExitFace that hold the ray coordinates in the entrance and exit face of object space for the central lenslet
# All three arrays are linear arrays whose elements are in the same sequence, so that camPix(n) is the pixel index of the ray the enters and exits object space at rayEntrFace(n) and rayExitFace(n)
# camRayCoord(voxCtr) accepts any voxCtr and generates the correct ray coordinates in the entrance and exit face associated with voxCtr

import numpy as np

nMedium = 1.33      #refractive index of object medium
naObj = 1.2         #NA of objective lens; for naObj=1.2 (water imm. objective), the tilt angle of ray passing through edge of aperture is arcSin(1.2/1.33)=64°
rNA = 7.7           #radius of NA in aperture plane behind microlens in fraction of camera pixels; rNA=7.7 is the measured aperture disc radius for the water imm. objective lens, a 100µm uLens diameter and 6.5 µm camPix pitch (Orca Flash4)
nrCamPix = 16       #nrCamPix is the number of camera pixels behind a lenslet in the horizontal and vertical direction
                    #0<=i<=nrCamPix and 0<=j<=nrCamPix span the plane of camera pixels, with i,j Reals
                    #integers [i,j] are the pixel count, with i and j starting at 0 and ending at nrCampix-1; [0.5,0.5] is the center location of the first pixel [0,0]
uLensCtr = [8.,8.]  #the µLens center is at the pixel border between 8th and 9th pixel, horizontally and vertically

voxCtr = (377/3,351.,351.)

# camPixRays generates a square list that holds values (in radian) for azimuth and tilt angles in object space for rays originating in camera pixels [i,j] behind a single lenslet
# The computation implements the sine condition for points in the back focal plane of the objective lens
def camPixRays(nrCamPix,uLensCtr,nMedium,naObj,rNA):
    angles=[[0.]*nrCamPix for i in range(nrCamPix)]   #creates a square list with nrCamPix * nrCamPix elements, each set to 0
    for i in range(nrCamPix):
        for j in range(nrCamPix):
            tmp = np.sqrt((i+0.5-uLensCtr[0])**2 + (j+0.5-uLensCtr[1])**2)
            if tmp <= rNA:
                angles[j][i]=[np.arctan2(i+0.5-uLensCtr[0],j+0.5-uLensCtr[1]),np.arcsin(tmp/rNA*naObj/nMedium)]
    return angles

# camRayCoord(voxCtr,angles) generates three arrays: camPix, rayEntrFace, rayExitFace.
# camRayCoord(voxCtr,angles) accepts any voxCtr and generates the correct ray coordinates in the entrance and exit face associated with voxCtr
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

# print for testing
angles=camPixRays(nrCamPix,uLensCtr,nMedium,naObj,rNA)
camPix,rayEntrFace,rayExitFace = camRayCoord(voxCtr,angles)
print(angles)
print(voxCtr)
print(len(rayEntrFace))
print(rayEntrFace)
print(len(rayExitFace))
print(rayExitFace)
print(len(camPix))
print(camPix)

'''
list_of_lists = [[["1,1,1","1,1,2","1,1,3"],["1,2,1","1,2,2","1,2,3"],["1,3,1","1,3,2","1,3,3"]],
                 [["2,1,1","2,1,2","2,1,3"],["2,2,1","2,2,2","2,2,3"],["2,3,1","2,3,2","2,3,3"]],
                 [["3,1,1","3,1,2","3,1,3"],["3,2,1","3,2,2","3,2,3"],["3,3,1","3,3,2","3,3,3"]]]

np_array = np.array(list_of_lists)
array_transposed = np.transpose(np_array)

print(np_array)
print("\n--------------\n\n",array_transposed)
'''

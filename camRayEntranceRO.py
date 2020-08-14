# 13 Aug 2020, RO
# The new camRayCoord(voxCtr) replaces the old camRayEntrance(voxCtr) function
# camRayCoord(voxCtr) function generates three arrays, as before, but now based on the text file camPixRays.txt
# camPixRays.txt holds the camPix indices and the azimuth and tilt angles of their associated rays in object space.
# The output of camrayCoord(voxCtr) are arrays camPix, rayEntrFace, rayExitFace.
# I renamed entrance and exit to rayEntrFace and rayExitFace, because I was wary of calling the array exit which shadows a system call exit
# camRayCoord(voxCtr) accepts any voxCtr and generates the correct ray coordinates in the entrance and exit face associated with voxCtr

import numpy as np

voxCtr = (377/3,351,351)

def camRayCoord(voxCtr):
    rays = []
    with open('camPixRays.txt', 'r') as f:
        for s in f:
            for rep in (('{Null}','0'),('{', '['), ('}', ']')): s = s.replace(rep[0], rep[1])
            x = eval(s)
            rays.append(x)
    camPix = []
    rayEntrFace = []
    rayExitFace = []
    for i in range(len(rays)):
        for j in range(len(rays[i])):
            if rays[i][j]!=0 :
                camPix.append([i,j])
                tmp=[voxCtr[0]*np.tan(rays[i][j][1]*np.pi/180)*np.sin(rays[i][j][0]*np.pi/180),
                     voxCtr[0]*np.tan(rays[i][j][1]*np.pi/180)*np.cos(rays[i][j][0]*np.pi/180)]
                rayEntrFace.append([0,voxCtr[1]+tmp[0],voxCtr[2]+tmp[1]])
                rayExitFace.append([2*voxCtr[0],voxCtr[1]-tmp[0],voxCtr[2]-tmp[1]])
    return camPix,rayEntrFace,rayExitFace

camPix,rayEntrFace,rayExitFace = camRayCoord(voxCtr)
print(voxCtr)
print(len(rayEntrFace))
print(rayEntrFace)
print(len(rayExitFace))
print(rayExitFace)
print(len(camPix))
print(camPix)


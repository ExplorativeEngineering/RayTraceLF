# camRayEntrance

"""voxCtr is the center of the voxel that is located in the center of object space, 
in other words voxCtr is the center of object space."""

#  position in object space given in µm ranges from {0, 0, 0} at one corner
#  to 2 * voxCtr at the diagonal corner
import numpy as np
import matplotlib.pyplot as plt
import LFRayTraceVoxParams
from LFRayTraceVoxSpace import getVoxelDims
from camRayEntranceRO import camRayCoord

"""
July 30 or so...
I seem to remember that the Siddon code could only deal with positive values as coordinates. 
That’s why I put the origin in one corner of the working space and then rastered it using 
the different voxPitch values. That’s why voxCtr is slightly different for different voxPitch values. 

At this point, I recommend sticking with this approach, even though the camRayEntrance coordinates need to 
slightly change for each voxPitch. To get around this, you can use the camRayEntrance coordinates 
that I have sent for the voxPitch = 25/16, subtract voxCtr [125.667, 351.000, 351.000] from each 
coordinate entry in camRayEntrance and then add the voxCtr for the current voxPitch. 
The result should be camRayEntrance coordinates that are corrected for the new voxCtr. 
"""

""" For the current optical setup (60x/1.2NA objective, Hamamatsu Flash4 camera), 
there are 164 camera pixels exposed to light and each of which receives a ray 
that has gone through object space. The list in file camRayEntrance.txt has 
164 elements, where each element is composed of the pixel location {i, j} on 
the camera and the location {0, y[i,j], z[I,j] } where the ray associated 
with {i, j} enters object space."""

# camRayEntrance -
#   Locations of rays where they pass through the entrance face
#   into object space.  Proved in file camRayEntrance.txt.
#
#   camRayEntrance[[1]] = {{2., 6.}, {0., 269.23612015234846, 139.08124584061335}}
#   ...
#   camRayEntrance[[164]] = {{15., 11.}, {0., 431.4138798476515, 561.5687541593866}}

# The entrance face to object space is characterized by x = 0 for all rays.
# For the exit face, x = 2 * voxCtr[[1]], where voxCtr[[1]] is the x-component of voxCtr.
# The y- and z-components of the exit face location of ray {i, j} can be constructed
# with voxCtr and the entrance face coordinates {0, y{i,j}, z{i,j}}.
# The y- and z-components of the exit face locations are
# 2 * voxCtr[[2]] - y{I,j} and 2 * voxCtr[[3]] - z{I,j}, respectively.



# def camRayEntrance(voxCtr):
#     # imports camRayEntrance and camPix from file
#     rays = []
#     with open('camRayEntrance26Div15.txt', 'r') as f:
#         for s in f:
#             # s = '{{2., 6.}, {0., 269.23612015234846, 139.08124584061335}}'
#             for rep in (('{', '['), ('}', ']')): s = s.replace(rep[0], rep[1])
#             x = eval(s)
#             rays.append(x)
#
#     camPix = []
#     entrance = []
#     exit = []
#     for i in range(len(rays)):
#         camPix.append(rays[i][0])
#         entrance.append(rays[i][1])
#         # ray[i][1][0] is zero
#         # ex = [rays[i][1][0] + 2 * voxCtr[0],
#         #        rays[i][1][1] + 2 * (voxCtr[1] - rays[i][1][1]),
#         #        rays[i][1][2] + 2 * (voxCtr[2] - rays[i][1][2])]
#         ex = [2 * voxCtr[0],
#               2 * voxCtr[1] - rays[i][1][1],
#               2 * voxCtr[2] - rays[i][1][2]]
#             # x = 2 * voxCtr[[1]], where voxCtr[[1]] is the x-component of voxCtr.
#             # y and z, 2 * voxCtr[[2]] - y{I,j} and 2 * voxCtr[[3]] - z{I,j}, respectively.
#         exit.append(ex)
        # TODO
    # print("CamPix, Entrance, Exit =========================== ")
    # for i in range(len(rays)):
    #     print(camPix[i], entrance[i], exit[i], end=" ")
    #     print()
    # print("end.")
    # return camPix, entrance, exit


    # TODO add angles here (as unit vectors, (x,y,z)
    #   Angles are x, y, z components, rather than angles(incl, azim)

    # List of 3 unit vectors, first is in the ray direction, second and third span the plane
    # perpendicular to the ray direction and are parallel to the y- and z-axis after rotating
    # the ray parallel to x-axis towards camera.
    # All three unit vectors are orthogonal to each other
    # rayUnitVectors[[rayNr]]={{rayDirX, rayDirY, rayDirZ}, {rayPolYX, rayPolYY, rayPolYZ}, {rayPolZX, rayPolZY, rayPolZZ}}

        # rayUnitVectors = Table[{tmp = rayExit[[rayNr]] - rayEnter[[rayNr]];
        #         tmp2 = Inverse[RotationMatrix[{tmp, {1, 0, 0}}]];
        #     tmp / Norm[tmp], tmp2.{0, 1, 0}, tmp2.{0, 0, 1}}, {rayNr, Length[rayEnter]}];

if __name__ == '__main__':
    # test
    voxPitch = (26/15)
    print("voxPitch:", voxPitch)
    voxCtr, voxNrX, voxNrYZ = getVoxelDims(LFRayTraceVoxParams.entranceExitX, LFRayTraceVoxParams.entranceExitYZ, voxPitch)
    print("    voxCtr, voxNrX, voxNrYZ: ", LFRayTraceVoxParams.formatList(voxCtr), voxNrX, voxNrYZ, "  extent(",
          LFRayTraceVoxParams.entranceExitX,
          LFRayTraceVoxParams.entranceExitYZ, "microns )")
    camPix, rayEntrFace, rayExitFace = camRayCoord(voxCtr)  # 164 (x,y), (x, y, z) (x, y, z)
    # Diagnostic...
    # Plot Entrance --------------------------
    ent = np.array(rayEntrFace)
    y = ent[:, [1]]
    z = ent[:, [2]]
    #colors = (0, 0, 0)
    area = np.pi * 3

    plt.figure("Entrance Coordinates")
    plt.scatter(y, z, s=area, alpha=0.5)
    # plt.scatter(y, z, s=area, c=colors, alpha=0.5)
    plt.title('entrance')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.show()
    # Plot Exit --------------------------
    exi = np.array(rayExitFace)
    y = exi[:, [1]]
    z = exi[:, [2]]
    #colors = (0, 0, 0)
    area = np.pi * 3

    plt.figure("Exit Coordinates")
    plt.scatter(y, z, s=area, alpha=0.5)
    # plt.scatter(y, z, s=area, c=colors, alpha=0.5)
    plt.title('Exit')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.show()

    # Plot camPix ------------------------
    ent = np.array(camPix)
    y = ent[:, [0]]
    z = ent[:, [1]]
    area = np.pi * 3

    plt.figure("CamPix Coordinates")
    plt.scatter(y, z, s=area, alpha=0.5)
    # plt.scatter(y, z, s=area, c=colors, alpha=0.5)
    plt.title('camPix')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.show()
    print()
    print(voxCtr)
    print(len(rayEntrFace))
    print(rayEntrFace)
    print(len(rayExitFace))
    print(rayExitFace)
    print(len(camPix))
    print(camPix)

"""
prints
...
[15.0, 9.0] [0.0, 362.82016070867627, 529.0141231187597] [0, 337.8298392913237, 171.63587688124028] 
[15.0, 10.0] [0.0, 393.77149844970035, 538.512459857001] [0, 306.87850155029963, 162.137540142999] 
[15.0, 11.0] [0.0, 431.4138798476515, 561.5687541593866] [0, 269.23612015234846, 139.0812458406134]
"""
# TODO generate camRayEntrance...
"""
Graphics object
# The entrance face is where rays enter object space. The face is in the y-z plane
entranceFaceGraphics := {
  	Table[Line[{{0, 1, 0}*voxPitch*
      i, {0, 1, 0}*voxPitch*i + {0, 0, 1}*voxPitch*voxNrYZ}], {i, 0, 
    voxNrYZ}],
  	Table[Line[{{0, 0, 1}*voxPitch*
      i, {0, 0, 1}*voxPitch*i + {0, 1, 0}*voxPitch*voxNrYZ}], {i, 0, 
    voxNrYZ}]
  }
# The exit face is where rays exit object space. The face is in the y-z plane
exitFaceGraphics := {
  	Table[Line[{{0, 1, 0}*voxPitch*i + {1, 0, 0}*voxPitch*
       voxNrX, {0, 1, 0}*voxPitch*i + {0, 0, 1}*voxPitch*
       voxNrYZ + {1, 0, 0}*voxPitch*voxNrX}], {i, 0, voxNrYZ}],
  	Table[Line[{{0, 0, 1}*voxPitch*i + {1, 0, 0}*voxPitch*
       voxNrX, {0, 0, 1}*voxPitch*i + {0, 1, 0}*voxPitch*
       voxNrYZ + {1, 0, 0}*voxPitch*voxNrX}], {i, 0, voxNrYZ}]
  }

# object cube
objectCubeGraphics := {
  	Table[Line[{{0, 1, 0}*voxPitch*i + {1, 0, 0}*voxPitch*j, {0, 1, 0}*
       voxPitch*i + {0, 0, 1}*voxPitch*voxNrYZ + {1, 0, 0}*voxPitch*
       j}], {i, 0, voxNrYZ}, {j, voxNrX - 1}],
  	Table[Line[{{0, 0, 1}*voxPitch*i + {1, 0, 0}*voxPitch*j, {0, 0, 1}*
       voxPitch*i + {0, 1, 0}*voxPitch*voxNrYZ + {1, 0, 0}*voxPitch*
       j}], {i, 0, voxNrYZ}, {j, voxNrX - 1}],
  	Table[Line[{{0, 0, 1}*voxPitch*i + {0, 1, 0}*voxPitch*j, {0, 0, 1}*
       voxPitch*i + {0, 1, 0}*voxPitch*j + {1, 0, 0}*voxPitch*
       voxNrX}], {i, 0, voxNrYZ}, {j, 0, voxNrYZ}]
  }

# The list camPixRays[[i,j]] holds values for azimuth and tilt angles 
# in object space for rays originating in camera pixels {i,j} behind a single lenslet
# The computation implements the sine condition for points in the back focal plane of the objective lens
# Angle values need to be converted and rounded to integer degrees, 
# otherwise later the test for all members of mP and iL having equal 
# length fails for all voxPitch values. I don't know why!

camPixRays = 
  Table[ 
		If[ 
            (tmp = Sqrt[(i - 0.5 - uLensCtr[[1]])^2 + 
                        (j - 0.5 - uLensCtr[[2]])^2 ] ) > rNA,
            {Null},
    	    {ArcTan[ i - 0.5 - uLensCtr[[1]], 
					 j - 0.5 - uLensCtr[[2]] ] / Degree, 
					 ArcSin[tmp / rNA * naObj / nMedium] / Degree
            } // Round
        ],
        {i, nrCamPix}, {j, nrCamPix}
    ];

# 2D line grid representing camera pixel array behind one microlens

camPixels2D := {
  	Table[Line[{{1, 0} * i, {1, 0} * i + {0, 1} * nrCamPix}], 
		{i, 0, nrCamPix}],
		
  	Table[Line[{{0, 1} * i, {0, 1} * i + {1, 0} * nrCamPix}],
		{i, 0, nrCamPix}],
		
  	Table[Text[{i, j}, {(i - 0.5), (j - 0.5)}],
		{i, nrCamPix}, {j, nrCamPix}]
  }

# 2D line grid representing entrance face of rays that traverse object space
entranceFace2D := {
  	Table[Line[{{1, 0} * voxPitch * i, 
  	            {1, 0} * voxPitch * i + {0, 1} * voxPitch * voxNrYZ}],
  	    {i, 0, voxNrYZ}
  	],
  	Table[Line[{{0, 1} * voxPitch * i, 
  	            {0, 1} * voxPitch * i + {1, 0} * voxPitch * voxNrYZ}],
  	    {i, 0, voxNrYZ}
  	]
  }

# The camRayEntrance list holds the positions where camera rays pass
# through the entrance face into the object cube
# the x position of the entrance face is 0, camera rays all converge on voxCtr

camRayEntrance := 
 Flatten[Table[ 
			If[camPixRays[[i, j]] == {Null}, , 
			{   {i, j},
				{0, voxCtr[[1]] * Tan[camPixRays[[i, j, 2]] Degree] 
						Sin[camPixRays[[i, j, 1]] Degree] + voxCtr[[2]],
					voxCtr[[1]] * Tan[camPixRays[[i, j, 2]] Degree] 
						Cos[camPixRays[[i, j, 1]] Degree] + voxCtr[[3]]
				}
			}	
			],  
			{i, nrCamPix}, {j, nrCamPix}
		] // N, 1
	] // DeleteCases[Null]

# List of object space coordinates where rays enter object space. The x-position is 0
rayEnter = Table[camRayEntrance[[i, 2]], {i, Length[camRayEntrance]}];

# List of object space coordinates where rays exit object space. The x-position is voxPitch*voxNrX
rayExit = Table[rayEnter[[i]] + 2 (voxCtr - rayEnter[[i]]), {i, Length[rayEnter]}];

# List of 3 unit vectors,first is in the ray direction,second and # third span the plane perpendicular 
# to the ray direction and are parallel to the y-and z-axis after rotating the ray parallel to x-axis towards camera
# All three unit vectors are orthogonal to each other.

# rayUnitVectors[[rayNr]]={{rayDirX,rayDirY,rayDirZ},{rayPolYX, rayPolYY,rayPolYZ},{rayPolZX,rayPolZY,rayPolZZ}}
rayUnitVectors = 
  Table[{	tmp = rayExit[[rayNr]] - rayEnter[[rayNr]]; 
			tmp2 = Inverse[RotationMatrix[{tmp, {1, 0, 0}}]]; 
			tmp/Norm[tmp],
    		tmp2.{0, 1, 0}, tmp2.{0, 0, 1}
		},
		{rayNr, Length[rayEnter]}
    ];
    
# initialization of array of tomographic images behind microlens array (camArray)
camArray =  Table[0, {i, nrCamPix}, {j, nrCamPix}];
camArrayY = Table[0, {i, nrCamPix}, {j, nrCamPix}];
camArrayZ = Table[0, {i, nrCamPix}, {j, nrCamPix}];

voxPitchList = {voxPitch, voxPitch, voxPitch};
voxNrList = {voxNrX, voxNrYZ, voxNrYZ};
"""
# =====================================
# Spatial coordinates and dimensions...

from LFRayTraceVoxParams import formatList

displace = [0, 0, 0]
# Entrance, Exit planes are 700 x 700, 250 apart (um)
entranceExitX = 250  # microns
entranceExitYZ = 700  # microns

def getVoxelDims(extentOfSpaceX, extentOfSpaceYZ, voxPitch):
    # Voxel dimensions, extent...
    # voxPitch is the side length in micron of a cubic voxel in object space
    # and of a square cell of the entrance and exit face
    dx = voxPitch
    dy = voxPitch
    dz = voxPitch
    def is_odd(a):
        return bool(a - ((a >> 1) << 1))
    # voxNrX is the number of voxels along the x-axis side of the object cube.
    # An odd number will put the Center of a voxel in the center of object space if even, add 1
    voxNrX = round(extentOfSpaceX / voxPitch)
    if not is_odd(voxNrX):
        voxNrX = voxNrX + 1
    # voxNrYZ is the number of voxels along the y- and z-axis side of the object cube
    # An odd number will put the Center of a voxel in the center of object space
    voxNrYZ = round(extentOfSpaceYZ / voxPitch)
    if not is_odd(voxNrYZ):
        voxNrYZ = voxNrYZ + 1
    # voxCtr is the location in physical object space on which all camera rays converge
    # is a coordinate (not an index)'''
    voxCtr = [voxNrX * voxPitch / 2, voxNrYZ * voxPitch / 2, voxNrYZ * voxPitch / 2]
    # voxCtr is a member of each midpoints list
    # print("voxCtr: ", voxCtr, "voxNrX: ", voxNrX, "voxNrYZ: ", voxNrYZ)
    return voxCtr, voxNrX, voxNrYZ


def getWorkingBox(voxCtr, workingBoxDim, voxPitch, displace):
    # Bounding Box of Sample, displaced
    """displace = {0 voxPitch, 0 voxPitch, 0 voxPitch}; #displacement from voxCtr
    displace is the displacement vector that moves the center of the bounding box
    containing the simulated object away from the center of object space. A displacement
    along the X-axis will be important for simulating objects that are not in the nominal focal plane. """
    displace = [0 * voxPitch, 0 * voxPitch, 0 * voxPitch]
    ''' voxBoxNrs specify the corner positions of the displaced bounding box in terms of indices (not physical length)
      voxBoxNrs = {{Xmin, Xmax}, {Ymin, Ymax}, {Zmin, Zmax}}
      voxBoxNrs = Transpose[{voxCtr + displace - sample[[2]] / 2, voxCtr + displace + sample[[2]] / 2}]
               / voxPitch // Round;'''
    voxBoxX1 = int(round((voxCtr[0] + displace[0] - workingBoxDim[0] / 2) / voxPitch))
    voxBoxX2 = int(round((voxCtr[0] + displace[0] + workingBoxDim[0] / 2) / voxPitch))
    voxBoxY1 = int(round((voxCtr[1] + displace[1] - workingBoxDim[1] / 2) / voxPitch))
    voxBoxY2 = int(round((voxCtr[1] + displace[1] + workingBoxDim[1] / 2) / voxPitch))
    voxBoxZ1 = int(round((voxCtr[2] + displace[2] - workingBoxDim[2] / 2) / voxPitch))
    voxBoxZ2 = int(round((voxCtr[2] + displace[2] + workingBoxDim[2] / 2) / voxPitch))
    voxBox = [[voxBoxX1, voxBoxX2], [voxBoxY1, voxBoxY2], [voxBoxZ1, voxBoxZ2]]
    return voxBox

def getWorkingDims(voxCtr, ulenses, voxPitch):
    workingSpaceX = 100  # 100 microns
    workingSpaceYZ = ulenses * voxPitch  # microns; YZ size a function of the # of uLens
    workingBoxDim = [workingSpaceX, workingSpaceYZ, workingSpaceYZ]
    workingBox = getWorkingBox(voxCtr, workingBoxDim, voxPitch, displace)
    print("    workingBox:", workingBox)
    print("               ",
                    (workingBox[0][1] - workingBox[0][0]),
                    (workingBox[1][1] - workingBox[1][0]),
                    (workingBox[2][1] - workingBox[2][0]))
    return workingBox

# test...
if __name__ == "__main__":
    # test
    voxPitch = (26 / 15) / 5
    ulenses = 3
    print("voxPitch, ulenses:", voxPitch, ulenses)
    voxCtr, voxNrX, voxNrYZ = getVoxelDims(entranceExitX, entranceExitYZ, voxPitch)
    print("    voxCtr, voxNrX, voxNrY:", formatList(voxCtr), voxNrX, voxNrYZ)
    workingSpaceX = 100  # 100 microns
    workingSpaceYZ = ulenses * voxPitch  # microns; YZ size a function of the # of uLens
    workingBoxDim = [workingSpaceX, workingSpaceYZ, workingSpaceYZ]
    workingBox = getWorkingBox(voxCtr, workingBoxDim, voxPitch, displace)
    print("    workingBox:", workingBox)


    """
    OLD...
    def getSpaceDims(extentX_, extentYZ_, voxPitch_, displace_):
        ''' Voxel dimensions, extent...
        voxPitch is the side length in micron of a cubic voxel in object space
        and of a square cell of the entrance and exit face '''
        print("        spaceDims (microns), voxPitch: ", extentX_, extentYZ_, displace_, voxPitch_)
        def is_odd(a):
            return bool(a - ((a >> 1) << 1))
        '''voxNrX is the number of voxels along the x-axis side of the object cube. An
        odd number will put the Center of a voxel in the center of object space
        if even, add 1'''
        voxNrX = round(extentX_ / voxPitch_)
        if not is_odd(voxNrX):
            voxNrX = voxNrX + 1
        ''' voxNrYZ is the number of voxels along the y- and z-axis side of the object cube
        An odd number will put the Center of a voxel in the center of object space'''
        voxNrYZ = round(extentYZ_ / voxPitch_)
        if not is_odd(voxNrYZ):
            voxNrYZ = voxNrYZ + 1
        # center voxel
        ''' voxCtr is the location in object space on which all camera rays converge
        is a coordinate (not an index)'''
        voxCtr = [voxNrX * voxPitch_ / 2, voxNrYZ * voxPitch_ / 2, voxNrYZ * voxPitch_ / 2]
        # voxCtr is a member of each midpoints list
        '''displace = {0 voxPitch, 0 voxPitch, 0 voxPitch}; #displacement from voxCtr
        displace is the displacement vector that moves the center of the bounding box
        containing the simulated object away from the center of object space. A displacement
        along the X-axis will be important for simulating objects that are not in the nominal focal plane. '''
        displace_ = [0 * voxPitch_, 0 * voxPitch_, 0 * voxPitch_]
        # voxBoxNrs specify the corner positions of the displaced bounding box in terms of indices (not physical length)
        # voxBoxX1 = round((voxCtr[0] + displace[0] - extentX_ / 2) / voxPitch_)
        # voxBoxX2 = round((voxCtr[0] + displace[0] + extentX_/ 2) / voxPitch_)
        # voxBoxY1 = round((voxCtr[1] + displace[1] - extentYZ_ / 2) / voxPitch_)
        # voxBoxY2 = round((voxCtr[1] + displace[1] + extentYZ_ / 2) / voxPitch_)
        # voxBoxZ1 = round((voxCtr[2] + displace[2] - extentYZ_ / 2) / voxPitch_)
        # voxBoxZ2 = round((voxCtr[2] + displace[2] + extentYZ_ / 2) / voxPitch_)
        # voxBox = [[voxBoxX1, voxBoxX2], [voxBoxY1, voxBoxY2], [voxBoxZ1, voxBoxZ2]]
        # TODO Forced to Zero... ?
        if voxNrYZ > 1:
            voxNYZ= voxNrYZ - 1
        else:
            voxNYZ = 1
        voxBox = [[0, voxNrX], [0, voxNYZ], [0, voxNYZ]]
        return voxNrX, voxNrYZ, voxBox

def getSpatialDims(ulenses, voxPitch):
    # obj_voxNrX, obj_voxNrYZ, ex_voxBox, ex_obj_offsets = getSpatialDims(voxPitch)
    ex_voxNrX, ex_voxNrYZ, ex_voxBox = getSpaceDims(entranceExitX, entranceExitYZ, voxPitch, displace)  # Voxel space
    print("    EntranceExitSpace: ", ex_voxNrX, ex_voxNrYZ, ex_voxBox)
    ex_voxBox = [[5, 140], [100, 300], [100, 300]]
    # Object space =========================
    workingSpaceX = 100  # 100 microns
    # YZ size a function of the # of uLens
    workingSpaceYZ = ulenses * voxPitch  # microns
    workingBoxDim=[]
    # WRONG? objectSpaceYZ = ulenses * 26 / 15  # microns
    obj_voxNrX, obj_voxNrYZ, obj_voxBox = getSpaceDims(objectSpaceX, objectSpaceYZ, voxPitch, displace)  # Voxel space
    print("    Object space:      ", obj_voxNrX, obj_voxNrYZ, obj_voxBox)
    # Coordinate transform --- object space coordinates relative to entrance/exit planes/volume
    # ex_obj_offsets = coordTranform(ex_voxBox, obj_voxBox)
    # for the calculation of the LFRayVoxelSpace, object space is placed in the center of entranceExitSpace
    # Sample is placed in object space coords
    offsetX = (ex_voxNrX - obj_voxNrX) /2
    offsetY = (ex_voxNrYZ - obj_voxNrYZ) / 2
    offsetZ = (ex_voxNrYZ - obj_voxNrYZ) / 2
    # offsetX = (ex_voxBox[0][1] - ex_voxBox[0][0] + 1) / 2 - (obj_voxBox[0][1] - obj_voxBox[0][0] + 1) / 2
    # offsetY = (ex_voxBox[1][1] - ex_voxBox[1][0] + 1) / 2 - (obj_voxBox[1][1] - obj_voxBox[1][0] + 1) / 2
    # offsetZ = (ex_voxBox[2][1] - ex_voxBox[2][0] + 1) / 2 - (obj_voxBox[2][1] - obj_voxBox[2][0] + 1) / 2
    ex_obj_offsets = [int(offsetX), int(offsetY), int(offsetZ)]
    return obj_voxNrX, obj_voxNrYZ, ex_voxBox, ex_obj_offsets

    """

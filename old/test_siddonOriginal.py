

from old.siddonOriginal import *
#import ipympl
import matplotlib.pyplot as plt

def test(*args, n):
    # Test Siddon
    print(args)
    midpts = midpoints(*args)
    lengths = intersect_length(*args)
    print(len(midpts),  'Midpoints: ' + str([(round(x[0], 3), round(x[1], 3), round(x[2], 3)) for x in midpts]))
    print(len(lengths), 'Lengths:   ' + str([round(x, 3) for x in lengths]))

    # Check lengths
    from math import sqrt
    print('Input length : ' + str(sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)))
    print('Output length: ' + str(sum(lengths)))
    print()

    """Plot"""
    fig = plt.figure("siddon 3d plot" + str(n))
    ax = plt.axes(projection='3d')
    ax.set_title('ray trace plot');

    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis');

    # Data for a three-dimensional line
    xline = [x[0] for x in midpts]
    yline = [x[1] for x in midpts]
    zline = [x[2] for x in midpts]
    ax.plot3D(xline, yline, zline, 'gray')

    # Data for three-dimensional scattered points
    xdata = [x[0] for x in midpts]
    ydata = [x[1] for x in midpts]
    zdata = [x[2] for x in midpts]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');


# Assemble arguments

x1, y1, z1 = 0.3, 0.4, 0.5
x2, y2, z2 = 1, 1, 1
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 10, 10, 10
args = (x1, y1, z1, x2, y2, z2, dx, dy, dz, nx, ny, nz)
test(*args,n=.9)

x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 1, 1, 1
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 10, 10, 10
args = (x1, y1, z1, x2, y2, z2, dx, dy, dz, nx, ny, nz)
test(*args,n=1)

x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 2, 2, 2
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 10, 10, 10
args = (x1, y1, z1, x2, y2, z2, dx, dy, dz, nx, ny, nz)
test(*args, n=2)

x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 2, 2, 2
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 20, 20, 20
args = (x1, y1, z1, x2, y2, z2, dx, dy, dz, nx, ny, nz)
test(*args, n=3)

x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 20, 20, 20
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 100, 100, 100
args = (x1, y1, z1, x2, y2, z2, dx, dy, dz, nx, ny, nz)
test(*args, n=4)

x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 20, 20, 20
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 20, 20, 20
args = (x1, y1, z1, x2, y2, z2, dx, dy, dz, nx, ny, nz)
test(*args, n=4.5)

x1, y1, z1 = 1, 1, 1
x2, y2, z2 = 20, 20, 20
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 10, 10, 10
args = (x1, y1, z1, x2, y2, z2, dx, dy, dz, nx, ny, nz)
test(*args, n=5)

x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 20, 20, 20
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 10, 10, 10
args = (x1, y1, z1, x2, y2, z2, dx, dy, dz, nx, ny, nz)
test(*args, n=6)



"""
I've attached an archive with code and a paper demonstrating how to pass
a ray through a field of voxels and split the ray into "voxel pieces"
that addresses the issues you ran into with the sphere and chord
approach.

Nathanial (my office mate) kindly offered to share his python
implementation. I added an example run in the test-siddon_original.py file. If
you run

python test-siddon_original.py

You'll run an example with the following parameters:

x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 1, 1, 1
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 10, 10, 10

A corner-to-corner ray of a unit cube split into 10x10x10 isometric
voxels. The output will be

Midpoints: [(0.05, 0.05, 0.05), (0.15, 0.15, 0.15), (0.25, 0.25, 0.25),
(0.35, 0.35, 0.35), (0.45, 0.45, 0.45), (0.55, 0.55, 0.55), (0.65, 0.65,
0.65), (0.75, 0.75, 0.75), (0.85, 0.85, 0.85), (0.95, 0.95, 0.95)]
Lengths: [0.173, 0.173, 0.173, 0.173, 0.173, 0.173, 0.173, 0.173, 0.173,
0.173]
Input length: 1.7320508075688772
Output length: 1.7320508075688772

You can modify test-siddon_original.py to try different rays. Another example:

x1, y1, z1 = 0.3, 0.4, 0.5
x2, y2, z2 = 1, 1, 1
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 10, 10, 10

Gives:

Midpoints: [(0.35, 0.443, 0.536), (0.408, 0.493, 0.577), (0.428, 0.51,
0.592), (0.47, 0.546, 0.621), (0.517, 0.586, 0.655), (0.557, 0.62,
0.683), (0.59, 0.649, 0.707), (0.625, 0.679, 0.732), (0.675, 0.721,
0.768), (0.71, 0.751, 0.793), (0.743, 0.78, 0.817), (0.783, 0.814,
0.845), (0.83, 0.854, 0.879), (0.872, 0.89, 0.908), (0.892, 0.907,
0.923), (0.95, 0.957, 0.964)]
Lengths: [0.15, 0.025, 0.035, 0.09, 0.05, 0.07, 0.03, 0.075, 0.075,
0.03, 0.07, 0.05, 0.09, 0.035, 0.025, 0.15]
Input length: 1.0488088481701514
Output length: 1.0488088481701514
"""
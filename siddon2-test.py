from siddon2 import midpoints, intersect_length, raytrace

# Assemble arguments
"""
x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 1, 1, 1
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 10, 10, 10

x1, y1, z1 = 0.3, 0.4, 0.5
x2, y2, z2 = 1, 1, 1
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 10, 10, 10

x1, y1, z1 = 0.3, 0.4, 0.5
x2, y2, z2 = 10, 10, 10
dx, dy, dz = 0.1, 0.1, 0.1
#nx, ny, nz = 10, 10, 10
pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut = 0, 10, 0, 10, 0, 10

x1, y1, z1 = 1, 1, 10
x2, y2, z2 = 6., 7., 8.
dx, dy, dz = 1.5, 1.5, 1.5
pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut = 0, 10, 0, 10, 0, 10


# Works...
x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 2.5, 2.5, 2.5
dx, dy, dz = 0.1, 0.1, 0.1
#nx, ny, nz = 10, 10, 10
#pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut = 0, 10, 0, 10, 0, 10
pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut = 0, 25, 0, 25, 0, 25
"""



x1, y1, z1 = 0.0, 281.03366217675773, 281.03366217675773
# 0, 162, 162
x2, y2, z2 = 50.85, 452.6729317038439, 430.2879679429044
# 29, 261, 248
dx, dy, dz = 1.73, 1.73, 1.73

#pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut = 65, 80, 195, 210, 195, 210
# 112, 138, 337, 363, 337, 363,

pix_numXIn, pix_numXOut, pix_numYIn, pix_numYOut, pix_numZIn, pix_numZOut = 0, 145, 0, 405, 0, 405
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

midpoints = midpoints(*args, alist)
lengths = intersect_length(*args, alist)
print('Midpoints: ' + str([(round(x[0], 3), round(x[1], 3), round(x[2], 3)) for x in midpoints]))
print('Lengths: ' + str([round(x, 3) for x in lengths]))

# Check lengths
from math import sqrt
print("#midpoints: ",len(midpoints))
print("#lengths: ",len(lengths))
print('Input length: ' + str(sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)))
print('Output length: ' + str(sum(lengths)))

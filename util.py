import numpy
import time
import pyglet

import noise
from config import SECTOR_SIZE

cb_v = numpy.array([
        [-1,+1,-1, -1,+1,+1, +1,+1,+1, +1,+1,-1],  # top
        [-1,-1,-1, +1,-1,-1, +1,-1,+1, -1,-1,+1],  # bottom
        [-1,-1,-1, -1,-1,+1, -1,+1,+1, -1,+1,-1],  # left
        [+1,-1,+1, +1,-1,-1, +1,+1,-1, +1,+1,+1],  # right
        [-1,-1,+1, +1,-1,+1, +1,+1,+1, -1,+1,+1],  # front
        [+1,-1,-1, -1,-1,-1, -1,+1,-1, +1,+1,-1],  # back
],dtype = numpy.float32)

def cube_v(pos,n):
    return n*cb_v+numpy.tile(pos,4)
def cube_v2(pos,n):
    return (n*cb_v)+numpy.tile(pos,4)[:,numpy.newaxis,:]

def cube_vertices(x, y, z, n):
    """ Return the vertices of the cube at position x, y, z with size 2*n.

    """
    return [
        [x-n,y+n,z-n, x-n,y+n,z+n, x+n,y+n,z+n, x+n,y+n,z-n],  # top
        [x-n,y-n,z-n, x+n,y-n,z-n, x+n,y-n,z+n, x-n,y-n,z+n],  # bottom
        [x-n,y-n,z-n, x-n,y-n,z+n, x-n,y+n,z+n, x-n,y+n,z-n],  # left
        [x+n,y-n,z+n, x+n,y-n,z-n, x+n,y+n,z-n, x+n,y+n,z+n],  # right
        [x-n,y-n,z+n, x+n,y-n,z+n, x+n,y+n,z+n, x-n,y+n,z+n],  # front
        [x+n,y-n,z-n, x-n,y-n,z-n, x-n,y+n,z-n, x+n,y+n,z-n],  # back
    ]


def tex_coord(x, y, n=4):
    """ Return the bounding vertices of the texture square.

    """
    m = 1.0 / n
    dx = x * m
    dy = y * m
    return [dx, dy, dx + m, dy, dx + m, dy + m, dx, dy + m]


def tex_coords(top, bottom, side):
    """ Return a list of the texture squares for the top, bottom and side.

    """
    top = tex_coord(*top)
    bottom = tex_coord(*bottom)
    side = tex_coord(*side)
    result = []
    result.append(top)
    result.append(bottom)
    for x in range(4):
        result.append(side)
    return result

FACES = [
    ( 0, 1, 0), #up
    ( 0,-1, 0), #down
    (-1, 0, 0), #left
    ( 1, 0, 0), #right
    ( 0, 0, 1), #forward
    ( 0, 0,-1), #back
]

noisen = noise.SimplexNoise(seed=int(time.time()))


def normalize(position):
    """ Accepts `position` of arbitrary precision and returns the block
    containing that position.

    Parameters
    ----------
    position : tuple of len 3

    Returns
    -------
    block_position : tuple of ints of len 3

    """
    x, y, z = position
    x, y, z = (int(round(x)), int(round(y)), int(round(z)))
    return (x, y, z)


def sectorize(position):
    """ Returns a tuple representing the sector for the given `position`.

    Parameters
    ----------
    position : tuple of len 3

    Returns
    -------
    sector : tuple of len 3

    """
    x, y, z = normalize(position)
    x, y, z = x / SECTOR_SIZE, y / SECTOR_SIZE, z / SECTOR_SIZE
    return (x*SECTOR_SIZE, 0, z*SECTOR_SIZE)

##monkey patch IndirectArrayRegion.__setitem__ to make it a bit quick for numpy arrays
orig_indirect_array_region_setitem = pyglet.graphics.vertexbuffer.IndirectArrayRegion.__setitem__
def numpy__setitem__(self, index, value):
    if isinstance(value, numpy.ndarray) and isinstance(index, slice) \
          and index.start is None and index.stop is None and index.step is None:
        arr = numpy.ctypeslib.as_array(self.region.array)
        for i in range(self.count):
            arr[i::self.stride] = value[i::self.count]
        return
    orig_indirect_array_region_setitem(self, index, value)
pyglet.graphics.vertexbuffer.IndirectArrayRegion.__setitem__ = numpy__setitem__

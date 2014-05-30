import numpy
import time
import pyglet
import pyglet.graphics as gl

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

c = 1
cb_v_half = numpy.array([
        [-1,+0,-1, -1,+0,+1, +1,+0,+1, +1,+0,-1],  # top
        [-1,-1,-1, +1,-1,-1, +1,-1,+1, -1,-1,+1],  # bottom
        [-c,-1,-1, -c,-1,+1, -c,+1,+1, -c,+1,-1],  # left
        [+c,-1,+1, +c,-1,-1, +c,+1,-1, +c,+1,+1],  # right
        [-1,-1,+c, +1,-1,+c, +1,+1,+c, -1,+1,+c],  # front
        [+1,-1,-c, -1,-1,-c, -1,+1,-c, +1,+1,-c],  # back
],dtype = numpy.float32)

c = 14.0/16
cb_v_cake = numpy.array([
        [-1,+0,-1, -1,+0,+1, +1,+0,+1, +1,+0,-1],  # top
        [-1,-1,-1, +1,-1,-1, +1,-1,+1, -1,-1,+1],  # bottom
        [-c,-1,-1, -c,-1,+1, -c,+1,+1, -c,+1,-1],  # left
        [+c,-1,+1, +c,-1,-1, +c,+1,-1, +c,+1,+1],  # right
        [-1,-1,+c, +1,-1,+c, +1,+1,+c, -1,+1,+c],  # front
        [+1,-1,-c, -1,-1,-c, -1,+1,-c, +1,+1,-c],  # back
],dtype = numpy.float32)

de_v = numpy.array([
        [0]*12,
        [0]*12,
        [-1,-1,+1, +1,-1,-1, +1,+1,-1, -1,+1,+1], 
        [+1,-1,-1, -1,-1,+1, -1,+1,+1, +1,+1,-1],
        [-1,-1,-1, +1,-1,+1, +1,+1,+1, -1,+1,-1], 
        [+1,-1,+1, -1,-1,-1, -1,+1,-1, +1,+1,+1],
],dtype = numpy.float32)

def cube_v(pos,n):
    return n*cb_v+numpy.tile(pos,4)
def cube_v2(pos,n):
    return (n*cb_v)+numpy.tile(pos,4)[:,numpy.newaxis,:]

def deco_v(pos,n):
    return n*de_v+numpy.tile(pos,4)
def deco_v2(pos,n):
    return (n*de_v)+numpy.tile(pos,4)[:,numpy.newaxis,:]


def tex_coord(x, y, n=4):
    """ Return the bounding vertices of the texture square.

    """
    m = 1.0 / n
    dx = x * m
    dy = y * m
    return [dx, dy, dx + m, dy, dx + m, dy + m, dx, dy + m]


def tex_coords(*sides): #top, bottom, 
    """ Return a list of the texture squares for the top, bottom and side.

    """
#    top = tex_coord(*top)
#    bottom = tex_coord(*bottom)
    result = []
#    result.append(top)
#    result.append(bottom)
    i=6
    for s in sides:
        result.append(tex_coord(*s))
        i-=1
    while i>=0:
        result.append(tex_coord(*sides[-1]))
        i-=1
    return result

FACES = [
    ( 0, 1, 0), #up
    ( 0,-1, 0), #down
    (-1, 0, 0), #left
    ( 1, 0, 0), #right
    ( 0, 0, 1), #forward
    ( 0, 0,-1), #back
]

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


class LineDrawGroup(pyglet.graphics.Group):
    def __init__(self, thickness = 1, parent=None):
        pyglet.graphics.Group.__init__(self, parent)
        self.thickness = thickness
    
    def set_state(self):
        gl.glLineWidth(self.thickness)
        gl.glColor3d(0, 0, 0)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        
    def unset_state(self):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glLineWidth(1)
        
class DrawTranslateGroup(pyglet.graphics.Group):
    def __init__(self, translate = (0,0,0), parent=None):
        pyglet.graphics.Group.__init__(self, parent)
        self.translate = translate
    
    def set_state(self):
        gl.glPushMatrix()
        gl.glTranslatef(*self.translate)
        
    def unset_state(self):
        gl.glPopMatrix()
        
class InventoryGroup(pyglet.graphics.Group):
    def __init__(self, parent=None):
        pyglet.graphics.Group.__init__(self, parent)
    
    def set_state(self):
        gl.glPushMatrix()
        gl.glTranslatef(0, 0, -64)
        gl.glRotatef(45, 1, 0, 0)
        gl.glRotatef(45, 0, 1, 0)
        
    def unset_state(self):
        gl.glPopMatrix()

class InventoryOutlineGroup(pyglet.graphics.Group):
    def __init__(self, parent=None):
        pyglet.graphics.Group.__init__(self, parent)
    
    def set_state(self):
        gl.glPushMatrix()
        gl.glTranslatef(0, 0, -60)
        gl.glRotatef(45, 1, 0, 0)
        gl.glRotatef(45, 0, 1, 0)
        
    def unset_state(self):
        gl.glPopMatrix()

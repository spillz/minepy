import numpy
from util import tex_coords, FACES, cb_v, de_v

TEXTURE_PATH = 'texture_fv.png'

white = numpy.tile(numpy.array([255,255,255]),6*4).reshape(6,3*4)
grass_top = numpy.array([30,188,30]*4+[255,255,255]*5*4).reshape(6,3*4)


class Block(object):
    name = None
    coords = None
    solid = True
    colors = white
    texture_fn = tex_coords
    vertices = cb_v

class Decoration(object):
    vertices = de_v
    solid = False

class DirtWithGrass(Block):
    name = 'Grass'
    coords = ((0, 15), (2, 15), (3, 15))
    colors = grass_top

class Sand(Block):
    name = 'Sand'
    coords = ((2, 14), (2, 14), (2, 14))
    
class Brick(Block):
    name = 'Brick'
    coords = ((7, 15), (7, 15), (7, 15)) #3 brick
    
class Stone(Block):
    name = 'Stone'
    coords = ((0, 14), (0, 14), (0, 14))  #4 stone
    
class Wood(Block):
    name = 'Wood'
    coords = ((5, 14), (5, 14), (4, 14))  #5 wood

class Plank(Block):
    name = 'Plank'
    coords = ((4, 15), (4, 15), (4, 15))  #6 plank
    
class Pumpkin(Block):
    name = 'Pumpkin'
    coords = ((6, 9), (6, 8), (7, 8), (6,8)) 
    
class Rose(Decoration, Block):
    name = 'Rose'
    coords = ((12,15), (12,15), (12,15))
#    vertices = de_v
#    solid = False

i = 1
BLOCKS = [b for b in Block.__subclasses__() if b.name != None]
BLOCK_ID = {}
for x in BLOCKS:
    print x.name,i
    BLOCK_ID[x.name] = i
    i+=1
BLOCK_NORMALS = numpy.array(FACES)
BLOCK_COLORS = numpy.array([white] + [x.colors for x in BLOCKS])
BLOCK_TEXTURES = numpy.array([tex_coords((0,0),(0,0),(0,0))] + [tex_coords(*x.coords) for x in BLOCKS],dtype = numpy.float32)/4
BLOCK_VERTICES = numpy.array([cb_v]+[x.vertices for x in BLOCKS])
BLOCK_SOLID = numpy.array([False]+[x.solid for x in BLOCKS], dtype = numpy.uint8)

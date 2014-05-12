import numpy
from util import tex_coords, FACES

TEXTURE_PATH = 'texture_fv.png'

BLOCKS = numpy.array([
    # top, bottom, side
    tex_coords((0, 0), (0, 0), (0, 0)), #0 - air, not actually used
    tex_coords((0, 15), (2, 15), (3, 15)), #1 dirt w/ grass
    tex_coords((2, 14), (2, 14), (2, 14)), #2 sand
    tex_coords((7, 15), (7, 15), (7, 15)), #3 brick
    tex_coords((0, 14), (0, 14), (0, 14)),  #4 stone
    tex_coords((5, 14), (5, 14), (4, 14)),  #5 wood
    tex_coords((4, 15), (4, 15), (4, 15))  #6 plank
    ],dtype = numpy.float32)/4

white = numpy.tile(numpy.array([255,255,255]),6*4).reshape(6,3*4)
grass_top = numpy.array([30,188,30]*4+[255,255,255]*5*4).reshape(6,3*4)

BLOCK_NORMALS = numpy.array(FACES)

BLOCK_COLORS = numpy.array([
    white, #air
    grass_top, #dirt w/ grass
    white, #sand
    white, #brick
    white, #stone
    white, #wood
    white, #plank
    ])

#BLOCKS = numpy.array([
#    tex_coords((0, 0), (0, 0), (0, 0)), #0 - air, not actually used
#    tex_coords((1, 0), (0, 1), (0, 0)), #1 dirt w/ grass
#    tex_coords((1, 1), (1, 1), (1, 1)), #2 sand
#    tex_coords((2, 0), (2, 0), (2, 0)), #3 brick
#    tex_coords((2, 1), (2, 1), (2, 1))  #4 stone
#    ],dtype = numpy.float32)

GRASS = 1
SAND = 2
BRICK = 3
STONE = 4
WOOD = 5
PLANK = 6


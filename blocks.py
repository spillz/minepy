import numpy
from util import tex_coords

TEXTURE_PATH = 'texture.png'

BLOCKS = numpy.array([
    tex_coords((0, 0), (0, 0), (0, 0)), #0 - air, not actually used
    tex_coords((1, 0), (0, 1), (0, 0)), #1
    tex_coords((1, 1), (1, 1), (1, 1)), #2
    tex_coords((2, 0), (2, 0), (2, 0)), #3
    tex_coords((2, 1), (2, 1), (2, 1))  #4
    ],dtype = numpy.float32)

GRASS = 1
SAND = 2
BRICK = 3
STONE = 4


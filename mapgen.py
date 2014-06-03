#std/external libs
import time
import numpy

#local libs
from config import SECTOR_SIZE, SECTOR_HEIGHT, LOADED_SECTORS
from blocks import BLOCK_VERTICES, BLOCK_COLORS, BLOCK_NORMALS, BLOCK_TEXTURES, BLOCK_ID, BLOCK_SOLID, TEXTURE_PATH
import noise

STONE = BLOCK_ID['Stone']
SAND = BLOCK_ID['Sand']
GRASS = BLOCK_ID['Grass']


class SectorNoise2D(object):
    def __init__(self, seed, step = SECTOR_SIZE, step_offset = 0, scale = 1, offset = 0):
        self.noise = noise.SimplexNoise(seed = seed)
        self.seed = seed
        self.step = step
        self.scale = scale
        self.offset = offset
        Z = numpy.mgrid[-1:SECTOR_SIZE+1,-1:SECTOR_SIZE+1].T #overgenerate by one block in each direction
        shape = Z.shape
        self.Z = Z.reshape((shape[0]*shape[1],2))+step_offset

    def __call__(self, position):
        Z = self.Z + numpy.array([position[0],position[2]])
        N=self.noise.noise(Z/self.step)*self.scale + self.offset
        return N.reshape((SECTOR_SIZE+2,SECTOR_SIZE+2))

def initialize_map_generator(seed = None):
    global noise1, noise2, noise3, noise4
    HILL_STEP = 40.0
    HILL_SCALE = 5
    HILL_OFFSET = 5
    CONTINENTAL_STEP = 1500.0
    CONTINENTAL_SCALE = 40.0
    CONTINENTAL_OFFSET = 80
    GAIN_STEP = 3000.0
    GAIN_SCALE = 5
    GAIN_OFFSET = 5
    if seed == None:
        seed = int(time.time())
    noise1 = SectorNoise2D(seed = seed+12, step = HILL_STEP, step_offset = 30,  
        scale = HILL_SCALE, offset = HILL_OFFSET)
    noise2 = SectorNoise2D(seed = seed+16, step = HILL_STEP, step_offset = 900,
        scale = HILL_SCALE, offset = HILL_OFFSET)
    noise3 = SectorNoise2D(seed = seed+14, step = CONTINENTAL_STEP, step_offset = 531,
        scale = CONTINENTAL_SCALE, offset = CONTINENTAL_OFFSET)
    noise4 = SectorNoise2D(seed = seed+18, step = GAIN_STEP, step_offset = 8123,
        scale = GAIN_SCALE, offset = GAIN_OFFSET)

def generate_sector(position, sector, world):
    """ Initialize the sector by procedurally generating terrain using
    simplex noise.

    """
    N1=noise1(position)
    N2=noise2(position)
    N3=noise3(position)
    N4=noise4(position)

    N1 = N1*N4+N3
    N2 = N2+N3

    b = numpy.zeros((SECTOR_HEIGHT,SECTOR_SIZE + 2,SECTOR_SIZE + 2),dtype='u2')
    for y in range(SECTOR_HEIGHT):
        b[y] = ((y<N1-3)*STONE + (((y>=N1-3) & (y<N1))*GRASS))
        thresh = ((y>N3)*(y<N2)*(y>10))>0
        b[y] = b[y]*(1 - thresh) + SAND * thresh
    return b.swapaxes(0,1).swapaxes(0,2)

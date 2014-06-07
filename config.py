import math

DIST = 128

TICKS_PER_SEC = 60

# Size of sectors used to ease block loading.
SECTOR_SIZE = 32 #width and depth
SECTOR_HEIGHT = 256 #height of world
LOADED_SECTORS = DIST//SECTOR_SIZE + 1 #number of sections in (x,z) directions to load sectors for

WALKING_SPEED = 5
FLYING_SPEED = 15

GRAVITY = 20.0
MAX_JUMP_HEIGHT = 1.0 # About the height of a block.
# To derive the formula for calculating jump speed, first solve
#    v_t = v_0 + a * t
# for the time at which you achieve maximum height, where a is the acceleration
# due to gravity and v_t = 0. This gives:
#    t = - v_0 / a
# Use t and the desired MAX_JUMP_HEIGHT to solve for v_0 (jump speed) in
#    s = s_0 + v_0 * t + (a * t^2) / 2
JUMP_SPEED = math.sqrt(2 * GRAVITY * MAX_JUMP_HEIGHT)
TERMINAL_VELOCITY = 50

PLAYER_HEIGHT = 2

SERVER_IP = None
SERVER_PORT = 20226

LOADER_IP = 'localhost'
LOADER_PORT = 20230

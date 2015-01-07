'''
world_loader.py -- manages client side terrain generation, world data caching, and syncing of terrain with multi-player server
'''

# standard library imports
import math
import itertools
import time
import numpy
import select
try:
    import cPickle as pickle
except ImportError:
    import pickle
import multiprocessing.connection
import socket
import sys

# local imports
from config import SECTOR_SIZE, SECTOR_HEIGHT, LOADED_SECTORS, SERVER_IP, SERVER_PORT, LOADER_IP, LOADER_PORT
from util import normalize, sectorize, FACES, cube_v, cube_v2
from blocks import BLOCK_VERTICES, BLOCK_COLORS, BLOCK_NORMALS, BLOCK_TEXTURES, BLOCK_ID, BLOCK_SOLID, TEXTURE_PATH
import mapgen

import logging
logging.basicConfig(level = logging.INFO)
def loader_log(msg, *args):
    logging.log(logging.INFO, 'LOADER: '+msg, *args)

SECTOR_ARRAY = numpy.mgrid[:SECTOR_SIZE,:SECTOR_HEIGHT,:SECTOR_SIZE].T
SH = SECTOR_ARRAY.shape
SECTOR_GRID = SECTOR_ARRAY.reshape((SH[0]*SH[1]*SH[2],3))

class WorldLoader(object):
    def __init__(self, client_pipe, server_pipe):
        self.client_pipe = client_pipe
        self.server_pipe = server_pipe
        self.db = None
        if self.server_pipe is None:
            import world_db
            self.db = world_db.World()
            self.world_seed = self.db.get_seed()
        else:
            self.server_pipe.send(['l_get_seed',[]])
            msg, (self.world_seed,) = self.server_pipe.recv()
            assert(msg == 'l_seed')
        self.pos = None
        self.blocks = numpy.zeros((SECTOR_SIZE+2,SECTOR_HEIGHT,SECTOR_SIZE+2),dtype='u2') #blocks of a sector not the whole world
        self.vt_data = None #vertex data for the current sector
        self._loader_loop()

    def _loader_loop(self):
        '''
        Receives request for terrain and vertices for sectors of the map from the client
        will check server for changed blocks. Current implementation is pretty dumb because
        it blocks at each client and server request and expects a single task at a time.
        
        TODO: load light across sector boundaries (potentially across four sectors for each loaded sector)
        
        When a player moves, we can drop sectors out of range and add new ones. For example:
         ++++++
        -OOOOO+
        -OOOOO+
        -OOOOO+
        -OOOOO+
        ------
        0 unchanged
        - sectors to drop
        + sectors to add        
        '''
        import mapgen
        import select
        loader_log('loader loop started')
        cpipe = self.client_pipe
        spipe = self.server_pipe
        mapgen.initialize_map_generator(seed = self.world_seed)
        while True:
            try:
                msg, data = cpipe.recv()
                loader_log('received from client %s', msg)
            except:
                loader_log('unexpected error reading from client pipe, exiting')
                return
            if msg == 'quit':
                loader_log('terminated by client')
                return
            if msg == 'sector_blocks':
                self.pos = data[0]
                sector_block_delta = None
                if spipe is not None:
                    loader_log('sending block request to server')
                    spipe.send(('l_get_sector_blocks', [self.pos]))
                    loader_log('getting block response from server')
                    msg, data = spipe.recv()
                    loader_log('got block response from server')
                    assert(msg == 'l_sector_blocks_changed')
                    spos, sector_block_delta = data
                    assert(spos == self.pos)
                else:
                    sector_block_delta = self.db.get_sector_data(self.pos)
                self._initialize(self.pos, sector_block_delta)
                self._calc_vertex_data(self.pos)
                cpipe.send_bytes(pickle.dumps(['sector_blocks',[self.pos, self.blocks, self.vt_data]],-1))
            if msg == 'set_block':
                notify_server, pos, block_id, sector_data = data
                sector_result = []
                for spos, blocks in sector_data:
                    self.blocks = blocks
                    self.set_block(pos, spos, block_id)
                    self._calc_vertex_data(spos)
                    sector_result.append((spos, self.blocks, self.vt_data))
                cpipe.send_bytes(pickle.dumps(['sector_blocks2',sector_result],-1))
                if spipe is not None:
                    if notify_server:
                        spipe.send(('set_block', [pos, block_id]))
                else:
                    self.db.set_block(pos, block_id)

    def _initialize(self, position, sector_block_delta):
        """ Initialize the sector by procedurally generating terrain using
        simplex noise.

        """
        self.blocks = mapgen.generate_sector(position, None, None)
        if sector_block_delta is not None:
            for p in sector_block_delta:
                self.blocks[p] = sector_block_delta[p]


    def _calc_exposed_faces(self):
        #TODO: The 3D bitwise ops are slow
        air = BLOCK_SOLID[self.blocks] == 0
        tr = air*(self.blocks>0)
        tr = tr | air

        #computes the faces that have at least one face exposed
        #and, hence, need to be drawn
        exposed = numpy.zeros(air.shape,dtype=numpy.uint8)
        exposed[:,:-1,:] |= (tr[:,:-1,:] | air[:,1:,:])<<7 #up
        exposed[:,1:,:] |= (tr[:,1:,:] | air[:,:-1,:])<<6 #down
        exposed[1:,:,:] |= (tr[1:,:,:] | air[:-1,:,:])<<5 #left
        exposed[:-1,:,:] |= (tr[:-1,:,:] | air[1:,:,:])<<4 #right
        exposed[:,:,:-1] |= (tr[:,:,:-1] | air[:,:,1:])<<3 #forward
        exposed[:,:,1:] |= (tr[:,:,1:] | air[:,:,:-1])<<2 #back
        self.exposed = exposed*(self.blocks>0)

        #first compute the air blocks that are directly exposed
        #to light from above
        direct_lit = numpy.cumproduct(air[:,::-1,:], axis=1)[:,::-1,:]>0
        unlit = (~direct_lit) & tr #& (self.exposed>0)

        ##TODO: For more even light, calculate a separate light value for the 24 vertices not the 6 faces
        unlit_pos = SECTOR_ARRAY[unlit[1:-1,1:-1,1:-1]] + [1,1,1] ##TODO: FILTER OUT EDGES
        lit = numpy.array(direct_lit, dtype = numpy.float32)
        for i in range(16): #propagate light up to 16 blocks, light level falls at each step
            if unlit_pos.shape[0]==0:
                break
            #find all unlit blocks that neighbor a lit block and give them a light value
            z,y,x = unlit_pos.T ##TODO: don't really understand why axes need to be reversed
            for f in FACES:
                z1,y1,x1 = (unlit_pos+f).T ##TODO: don't really understand why axes need to be reversed
                b = lit[x1,y1,z1]*0.75
                a = lit[x,y,z]
                lit[x,y,z] = a + (a<b)*(b-a)
            #now filter to remaining unlit blocks
            unlit_pos = unlit_pos[lit[x,y,z] == 0]
        #now we mark the block faces that are exposed to a lit air block
        exposed_light = numpy.zeros(lit.shape+(6,),dtype=numpy.float32)
        exposed_light[:,:-1,:,0] = lit[:,1:,:] #up -- was 7
        exposed_light[:,1:,:,1] = lit[:,:-1,:] #down -- was 6
        exposed_light[1:,:,:,2] = lit[:-1,:,:] #left -- ...
        exposed_light[:-1,:,:,3] = lit[1:,:,:] #right
        exposed_light[:,:,:-1,4] = lit[:,:,1:] #forward
        exposed_light[:,:,1:,5] = lit[:,:,:-1] #back
        self.exposed_light = exposed_light

    def _calc_vertex_data(self,position):
        self._calc_exposed_faces()
        exposed = self.exposed[1:-1,:,1:-1]
        exposed_light = self.exposed_light[1:-1,:,1:-1]
        sh = exposed.shape
        exposed = exposed.swapaxes(0,2).reshape(sh[0]*sh[1]*sh[2]) ##TODO: don't really understand why axes need to be reversed
        exposed_light = exposed_light.swapaxes(0,2).reshape(sh[0]*sh[1]*sh[2],6)
        egz = exposed>0
        pos = SECTOR_GRID[egz] + position
        exposed = exposed[egz]
        exposed_light = exposed_light[egz]
        exposed = numpy.unpackbits(exposed[:,numpy.newaxis],axis=1)
        exposed = numpy.array(exposed,dtype=bool)
        exposed = exposed[:,:6]
        p = (pos - numpy.array(position)).T
        b = self.blocks[1:-1,:,1:-1][p[0],p[1],p[2]]
        texture_data = BLOCK_TEXTURES[b]
        color_data = numpy.array(BLOCK_COLORS[b]*(0.025 + 0.975*exposed_light[:,:,numpy.newaxis]),dtype = numpy.int32)
        normal_data = numpy.tile(BLOCK_NORMALS, (len(b),1,4))#*exposed_light[:,:,numpy.newaxis]
        vertex_data = 0.5*BLOCK_VERTICES[b] + numpy.tile(pos, 4)[:,numpy.newaxis,:]

        v = vertex_data[exposed].ravel()
        t = texture_data[exposed].ravel()
        n = normal_data[exposed].ravel()
        c = color_data[exposed].ravel()
        count = len(v)/3

        self.vt_data = (count, v, t, n, c)

    def get_block(self, position, sector_position):
        pos = position - numpy.array(sector_position)
        if len(pos.shape)>1:
            pos = pos.T
            return self.blocks[pos[0]+1,pos[1],pos[2]+1]
        return self.blocks[pos[0]+1,pos[1],pos[2]+1]

    def set_block(self, position, sector_position, val):
        pos = position - numpy.array(sector_position)
        if len(pos.shape)>1:
            pos = pos.T
            self.blocks[pos[0]+1, pos[1], pos[2]+1] = val
        self.blocks[pos[0]+1, pos[1], pos[2]+1] = val

##TODO: Move to world_proxy so that we don't need to import the module
##and its dependencies in the main client process (probably doesn't
##really matter on Linux because of the way that fork works)
def _start_loader(client_pipe, server_pipe):
    WorldLoader(client_pipe, server_pipe)

def start_loader(server_pipe = None):
    pipe, _pipe = multiprocessing.Pipe()
    process = multiprocessing.Process(target = _start_loader, args = (_pipe, server_pipe))
    process.start()
    return pipe


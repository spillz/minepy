# standard library imports
import math
import itertools
import threading
import time
import numpy
import multiprocessing.connection
import multiprocessing.sharedctypes
import multiprocessing
import cPickle

# pyglet imports
import pyglet
image = pyglet.image
from pyglet.graphics import TextureGroup
import pyglet.gl as gl

# local imports
import config
from config import SECTOR_SIZE, SECTOR_HEIGHT, LOADED_SECTORS
from util import normalize, sectorize, FACES, cube_v, cube_v2
from blocks import BLOCK_VERTICES, BLOCK_COLORS, BLOCK_NORMALS, BLOCK_TEXTURES, BLOCK_ID, BLOCK_SOLID, TEXTURE_PATH
import mapgen

SECTOR_GRID = numpy.mgrid[:SECTOR_SIZE,:SECTOR_HEIGHT,:SECTOR_SIZE].T
SH = SECTOR_GRID.shape
SECTOR_GRID = SECTOR_GRID.reshape((SH[0]*SH[1]*SH[2],3))

class SectorLoader(object):
    def __init__(self):
        self.pipe, self._pipe = multiprocessing.Pipe()
#        self.blocks = multiprocessing.Array('B',SECTOR_SIZE*SECTOR_SIZE*SECTOR_HEIGHT)
#        self.v = multiprocessing.Array('f',200000)
#        self.t = multiprocessing.Array('f',200000)
#        self.n = multiprocessing.Array('f',200000)
#        self.c = multiprocessing.Array('f',200000)
        self.blocks = numpy.zeros((SECTOR_SIZE,SECTOR_HEIGHT,SECTOR_SIZE),dtype='u2')
        self.vt_data = None

#        self.count = multiprocessing.Value('f')
        self.process = multiprocessing.Process(target = self._loader)
        self.process.start()

    def _loader(self):
        print('loader started')
        mapgen.initialize_map_generator()
        while True:
            try:
                msg, data = self._pipe.recv()
                print('received', msg, data)
            except:
                return
            if msg == 'quit':
                print('loader terminated by owner')
                return
            if msg == 'request_sector':
                pos = (data[0], -40, data[2])
                self._initialize(pos)
                self._calc_vertex_data(pos)
                self._pipe.send_bytes(cPickle.dumps([pos, self.blocks, self.vt_data],-1))

    def poll(self):
        return self.pipe.poll()

    def send(self, list_object):
        self.pipe.send(list_object)
        
    def send_bytes(self, bytes_object):
        self.pipe.send_bytes(bytes_object)

    def recv(self):
        return self.pipe.recv()
        
    def recv_bytes(self):
        return self.pipe.recv_bytes()

    def _initialize(self, position):
        """ Initialize the sector by procedurally generating terrain using
        simplex noise.

        """
        self.blocks = mapgen.generate_sector(position, None, None)

    def _calc_exposed_faces(self):
        #TODO: The 3D bitwise ops are slow
        t = time.time()
        air = BLOCK_SOLID[self.blocks] == 0

        light = numpy.cumproduct(air[:,::-1,:], axis=1)[:,::-1,:]

        exposed = numpy.zeros(air.shape,dtype=numpy.uint8)
#        exposed[0,:,:] |= self.edge_blocks(dx=-1)<<5 #left edge
#        exposed[-1,:,:] |= self.edge_blocks(dx=1)<<4 #right edge
#        exposed[:,:,0] |= self.edge_blocks(dz=-1)<<2 #back edge
#        exposed[:,:,-1] |= self.edge_blocks(dz=1)<<3 #front edge
        exposed[:,:-1,:] |= air[:,1:,:]<<7 #up
        exposed[:,1:,:] |= air[:,:-1,:]<<6 #down
        exposed[1:,:,:] |= air[:-1,:,:]<<5 #left
        exposed[:-1,:,:] |= air[1:,:,:]<<4 #right
        exposed[:,:,:-1] |= air[:,:,1:]<<3 #forward
        exposed[:,:,1:] |= air[:,:,:-1]<<2 #back
        self.exposed = exposed*(~air)

    def _calc_vertex_data(self,position):
        self._calc_exposed_faces()
        sh = self.exposed.shape
        exposed = self.exposed.swapaxes(0,2).reshape(sh[0]*sh[1]*sh[2])
        egz = exposed>0
        pos = SECTOR_GRID[egz] + position
        exposed = exposed[egz]
        exposed = numpy.unpackbits(exposed[:,numpy.newaxis],axis=1)
        exposed = numpy.array(exposed,dtype=bool)
        exposed = exposed[:,:6]
        #b = self[pos]
        p = (pos - numpy.array(position)).T
        b = self.blocks[p[0],p[1],p[2]]
        texture_data = BLOCK_TEXTURES[b]
        color_data = BLOCK_COLORS[b]
        normal_data = numpy.tile(BLOCK_NORMALS,(len(b),1,4))#*exposed_light[:,:,numpy.newaxis]
        vertex_data = 0.5*BLOCK_VERTICES[b] + numpy.tile(pos,4)[:,numpy.newaxis,:]

        v = vertex_data[exposed].ravel()
        t = texture_data[exposed].ravel()
        n = normal_data[exposed].ravel()
        c = color_data[exposed].ravel()
        count = len(v)/3

        self.vt_data = (count, v, t, n, c)
        
        

class SectorProxy(object):
    def __init__(self, position, group, model, shown=True):
        self.position = position[0],-40,position[2]
        self.group = group
        self.model = model
        # A Batch is a collection of vertex lists for batched rendering.
        self.batch = pyglet.graphics.Batch()
        self.blocks = numpy.zeros((SECTOR_SIZE,SECTOR_HEIGHT,SECTOR_SIZE),dtype='u2')
        self.shown = shown
        # Mapping from position to a pyglet `VertextList` for all shown blocks.
        self.vt = None
        self.vt_data = None
        self.invalidate_vt = False

    def draw(self, draw_invalid = True):
        if draw_invalid and self.invalidate_vt:
            self.check_show()
            self.invalidate_vt = False
            draw_invalid = False
        self.batch.draw()
        return draw_invalid

    def __getitem__(self, position):
#        position = normalize(position)
        pos = position - numpy.array(self.position)
        if len(pos.shape)>1:
            pos = pos.T
            return self.blocks[pos[0],pos[1],pos[2]]
        return self.blocks[pos[0],pos[1],pos[2]]

    def grid(self):
        grid = SECTOR_GRID + numpy.array(self.position)
        return grid

    def invalidate(self):
        self.invalidate_vt = True
        self.vt_data = None

    def check_show(self,add_to_batch = True):
        if add_to_batch and self.vt_data:
            (count, v, t, n, c) = self.vt_data
            self.vt = self.batch.add(count, gl.GL_QUADS, self.group,
                ('v3f/static', v),
                ('t2f/static', t),
                ('n3f/static', n),
                ('c3B/static', c))
            self.vt_data = None


class ModelProxy(object):

    def __init__(self):

        # A TextureGroup manages an OpenGL texture.
        self.group = TextureGroup(image.load(TEXTURE_PATH).get_texture())

        # The world is stored in sector chunks.
        self.sectors = {}
        self.sector_lock = threading.Lock()
        self.thread = None

        self.n_requests = 0
        self.n_responses = 0

        if config.SERVER_IP == None:
            self.loader = SectorLoader()
        else:
            self.loader = multiprocessing.connection.Client(address = (config.SERVER_IP,config.SERVER_PORT), authkey = 'password')

    def __getitem__(self, position):
        """
        retrieves the block at the (x,y,z) coordinate tuple `position`
        """
        try:
            return self.sectors[sectorize(position)][position]
        except:
            return None

    def add_block(self, position, block):
        pass
        
    def remove_block(self, position):
        pass
        
    def draw(self, position, (center, radius)):
        #t = time.time()
        draw_invalid = True
        with self.sector_lock:
            for s in self.sectors:
                spos = numpy.array([s[0], s[2]])+SECTOR_SIZE/2
                if ((center-spos)**2).sum()>(radius+SECTOR_SIZE/2)**2:
                    continue
                if self.sectors[s].shown:
                    draw_invalid = self.sectors[s].draw(draw_invalid)

    def neighbor_sectors(self, pos):
        """
        return a tuple (dx, dz, sector) of currently loaded neighbors to the sector at pos
        """
        pos = sectorize(pos)
        for x in ((-1,0),(1,0),(0,-1),(0,1)):
            npos = (pos[0]+x[0]*SECTOR_SIZE,0,pos[2]+x[1]*SECTOR_SIZE)
            if npos in self.sectors:
                yield x[0],x[1],self.sectors[npos]

    def update_sectors(self, old, new):
        """
        the observer has moved from sector old to new
        """
        if self.n_requests <= self.n_responses:
            new = sectorize(new)
            if old is not None:
                old = sectorize(old)
            if old != new:
                self.sectors_pos = []
                G = range(-LOADED_SECTORS,LOADED_SECTORS+1)
                for dx,dy,dz in itertools.product(G,(0,),G):
                    pos = numpy.array([new[0],new[1],new[2]]) \
                        + numpy.array([dx*SECTOR_SIZE,dy,dz*SECTOR_SIZE])
                    pos = sectorize(pos)
                    dist = (pos[0]-new[0])**2 + (pos[2]-new[2])**2
                    if pos not in self.sectors:
                        self.sectors_pos.append((dist,pos))
                for s in list(self.sectors):
                    if (new[0] - s[0])**2 + (new[2] - s[2])**2 > (LOADED_SECTORS*SECTOR_SIZE)**2:
                        print('dropping sector',s)
                        del self.sectors[s]
                self.sectors_pos = sorted(self.sectors_pos)
            if len(self.sectors_pos)>0:
                spos = self.sectors_pos.pop(0)[1]
                print 'requesting sector',spos
                self.loader.send(['request_sector',spos])
                self.n_requests += 1
        if self.loader.poll():
            spos, blocks, vt_data = self.loader.recv()
            self.n_responses+=1
            print 'recv',spos,len(vt_data[1])
            s = SectorProxy(spos,self.group,self)
            s.blocks[:,:,:] = blocks
            s.vt_data = vt_data
            self.sectors[sectorize(spos)] = s
            s.check_show()
            ##add the sector to the worldproxy

    def hit_test(self, position, vector, max_distance=8):
        """ Line of sight search from current position. If a block is
        intersected it is returned, along with the block previously in the line
        of sight. If no block is found, return None, None.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position to check visibility from.
        vector : tuple of len 3
            The line of sight vector.
        max_distance : int
            How many blocks away to search for a hit.

        """
        m = 8
        x, y, z = position
        dx, dy, dz = vector
        previous = None
        for _ in xrange(max_distance * m):
            key = normalize((x, y, z))
            if key != previous:
                b = self[key]
                if b != 0 and b is not None:
                    return key, previous
            previous = key
            x, y, z = x + dx / m, y + dy / m, z + dz / m
        return None, None

    def exposed(self, position):
        """ Returns False is given `position` is surrounded on all 6 sides by
        blocks, True otherwise.

        """
        x, y, z = position
        for dx, dy, dz in FACES:
            b = self[normalize((x + dx, y + dy, z + dz))]
            if not BLOCK_SOLID[b]:
                return True
        return False

    def quit(self,kill_server=True):
        if kill_server:
            self.loader.send(['quit',0])

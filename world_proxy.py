# standard library imports
import math
import itertools
import time
import numpy
import multiprocessing.connection
import multiprocessing.sharedctypes
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle
import gc

#gc.set_debug(gc.DEBUG_LEAK)

# pyglet imports
import pyglet
image = pyglet.image
from pyglet.graphics import TextureGroup
import pyglet.gl as gl

# local imports
import world_loader
import server_connection
import config
from config import SECTOR_SIZE, SECTOR_HEIGHT, LOADED_SECTORS
from util import normalize, sectorize, FACES, cube_v, cube_v2
from blocks import BLOCK_VERTICES, BLOCK_COLORS, BLOCK_NORMALS, BLOCK_TEXTURES, BLOCK_ID, BLOCK_SOLID, TEXTURE_PATH
import mapgen

#import logging
#logging.basicConfig(level = logging.INFO)
#def world_log(msg, *args):
#    logging.log(logging.INFO, 'WORLD: '+msg, *args)

class SectorProxy(object):
    def __init__(self, position, batch, group, model, shown=True):
        self.position = position[0],0,position[2]
        self.bposition = position[0]-1,0,position[2]-1 #block relative position (the sector is overgenerated by one block along each x,z edge)
        self.group = group
        self.model = model
        # A Batch is a collection of vertex lists for batched rendering.
        self.batch = batch
        self.blocks = numpy.zeros((SECTOR_SIZE+2,SECTOR_HEIGHT,SECTOR_SIZE+2),dtype='u2')
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
        pos = position - numpy.array(self.bposition)
        if len(pos.shape)>1:
            pos = pos.T
            return self.blocks[pos[0],pos[1],pos[2]]
        return self.blocks[pos[0],pos[1],pos[2]]

    def invalidate(self):
        self.invalidate_vt = True

    def check_show(self,add_to_batch = True):
        if add_to_batch and self.vt_data is not None:
            if self.vt != None:
                print('deleting vt',self.position)
                self.vt.delete()
            (count, v, t, n, c) = self.vt_data
            self.vt = self.batch.add(count, gl.GL_QUADS, self.group,
                ('v3f/static', v.copy()),
                ('t2f/static', t.copy()),
                ('n3f/static', n.copy()),
                ('c3B/static', c.copy()))
            self.vt_data = None


class ModelProxy(object):

    def __init__(self):

        # A TextureGroup manages an OpenGL texture.
        self.group = TextureGroup(image.load(TEXTURE_PATH).get_texture())
        self.unused_batches = []

        # The world is stored in sector chunks.
        self.sectors = {}
        self.update_sectors_pos = []
        self.update_ref_pos = None

        self.loader_requests = []
        self.active_loader_request = [None, None]
        self.n_requests = 0
        self.n_responses = 0

        loader_server_pipe = None
        self.server = None
        if config.SERVER_IP is not None:
            print ('Starting server on %s'%(config.SERVER_IP,))
            self.server = server_connection.start_server_connection(config.SERVER_IP)
            loader_server_pipe = self.server.loader_pipe
        print ('Starting sector loader')
        self.loader = world_loader.start_loader(loader_server_pipe)
    
    def get_batch(self):
        if len(self.unused_batches)>0:
            return self.unused_batches.pop(0)
        else:
            return pyglet.graphics.Batch()

    def release_sector(self, sector):
        if sector.vt is not None:
            sector.vt.delete()
        self.unused_batches.append(sector.batch)
        del self.sectors[sector.position]

    def __getitem__(self, position):
        """
        retrieves the block at the (x,y,z) coordinate tuple `position`
        """
        try:
            return self.sectors[sectorize(position)][position]
        except:
            return None

    def add_block(self, position, block, notify_server = True):
        spos = sectorize(position)
        if spos in self.sectors:
            s = self.sectors[spos]
            blocks = s.blocks
            nspos = None
            nblocks = None
            for np in [(1,0,0), (-1,0,0), (0,0,1), (0,0,-1)]:
                nspos = sectorize((position[0]+np[0],position[1]+np[1],position[2]+np[2]))
                if nspos != spos and nspos in self.sectors:
                    nblocks = self.sectors[nspos].blocks
                    break
            self.loader_requests.insert(0,['set_block', [notify_server, position, block, spos, blocks, nspos, nblocks]])

    def remove_block(self, position, notify_server = True):
        self.add_block(position, 0)

    def draw(self, position, (center, radius)):
        #t = time.time()
        draw_invalid = True
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
            if self.update_ref_pos != new:
                self.update_ref_pos = new
                self.update_sectors_pos = []
                G = range(-LOADED_SECTORS,LOADED_SECTORS+1)
                for dx,dy,dz in itertools.product(G,(0,),G):
                    pos = numpy.array([new[0],new[1],new[2]]) \
                        + numpy.array([dx*SECTOR_SIZE,dy,dz*SECTOR_SIZE])
                    pos = sectorize(pos)
                    dist = (pos[0]-new[0])**2 + (pos[2]-new[2])**2
                    if pos not in self.sectors and pos != self.active_loader_request[1]:
                        self.update_sectors_pos.append((dist,pos))
                for s in list(self.sectors):
                    if abs(new[0] - s[0])>LOADED_SECTORS*SECTOR_SIZE or abs(new[2] - s[2]) > LOADED_SECTORS*SECTOR_SIZE:
                        print('dropping sector',s,len(self.sectors))
                        self.release_sector(self.sectors[s])
                self.update_sectors_pos = sorted(self.update_sectors_pos)
            if len(self.update_sectors_pos)>0:
                self.active_loader_request = self.update_sectors_pos.pop(0)
                spos = self.active_loader_request[1]
                print('queueing sector',spos)
                try:
                    req_pos = [r[0] for r in self.loader_requests].index('sector_blocks') #insert job below higher priority jobs
                except ValueError:
                    req_pos = -1
                self.loader_requests.insert(req_pos,['sector_blocks',[spos]])
            if len(self.loader_requests)>0:
                self.loader_time = time.time()
                self.n_requests += 1
                print('client sending request to loader',self.loader_requests[0][0])
                self.loader.send(self.loader_requests.pop(0))

        if self.loader.poll():
            try:
                msg, data = self.loader.recv()
                print('client received',msg)
                if msg == 'sector_blocks':
                    spos1, b1, v1 = data
                    self.n_responses = self.n_requests
                    self.active_loader_request = [None, None]
                    print('took', time.time()-self.loader_time)
                    self._update_sector(spos1, b1, v1)
                if msg == 'sector_blocks2':
                    spos1, b1, v1, spos2, b2, v2 = data
                    self.n_responses = self.n_requests
                    self.active_loader_request = [None, None]
                    print('took', time.time()-self.loader_time)
                    self._update_sector(spos1, b1, v1)
                    self._update_sector(spos2, b2, v2)
            except EOFError:
                print('loader returned EOF')
        
        if self.server and self.server.poll():
            try:
                msg, data = self.server.recv()
                if msg == 'connected':
                    self.player, self.players = data
                if msg == 'player_set_block':
                    print data
                    pos, block = data
                    self.add_block(pos, block, False)
            except EOFError:
                print('server returned EOF')

    def _update_sector(self, spos, b, v):
        if b is not None:
            if spos in self.sectors:
                print('updating existing sector data',spos)
                s = self.sectors[spos]
                s.blocks[:,:,:] = b
                s.vt_data = v
                s.invalidate()
            else:
                print('setting new sector data',spos)
                s = SectorProxy(spos, self.get_batch(), self.group, self)
                s.blocks[:,:,:] = b
                s.vt_data = v
                self.sectors[sectorize(spos)] = s
                s.invalidate()

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
        if self.n_requests > self.n_responses:
            self.loader.recv()
        print('shutting down loader')
        self.loader.send(['quit',0])
        if self.server is not None:
            print('closing server connection')
            self.server.send(['quit',0])
                

'''
world_loader.py -- manages client side terrain generation, world data caching, syncing with multi-player server
'''

# standard library imports
import math
import itertools
import time
import numpy
import select
import cPickle
import multiprocessing.connection
import socket
import sys

# local imports
from config import SECTOR_SIZE, SECTOR_HEIGHT, LOADED_SECTORS, SERVER_IP, SERVER_PORT, LOADER_IP, LOADER_PORT
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
        self.blocks = numpy.zeros((SECTOR_SIZE+2,SECTOR_HEIGHT,SECTOR_SIZE+2),dtype='u2')
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
                print('received', msg)
            except:
                return
            if msg == 'quit':
                print('loader terminated by owner')
                return
            if msg == 'request_sector':
                pos = (data[0], 0, data[2])
                self._initialize(pos)
                self._calc_vertex_data(pos)
                self._pipe.send_bytes(cPickle.dumps([pos, self.blocks, self.vt_data],-1))
            if msg == 'add_block':
                spos = data[0]
                pos = data[1]
                block_id = data[2]
                self.blocks = data[3]
                self.set_block(pos, spos, block_id)
                self._calc_vertex_data(spos)
                self._pipe.send_bytes(cPickle.dumps([spos, self.blocks, self.vt_data],-1))
            if msg == 'remove_block':
                spos = data[0]
                pos = data[1]
                self.blocks = data[2]
                self.set_block(pos, spos, 0)
                self._calc_vertex_data(spos)
                self._pipe.send_bytes(cPickle.dumps([spos, self.blocks, self.vt_data],-1))

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
        air = BLOCK_SOLID[self.blocks] == 0

        light = numpy.cumproduct(air[:,::-1,:], axis=1)[:,::-1,:]

        exposed = numpy.zeros(air.shape,dtype=numpy.uint8)
        exposed[:,:-1,:] |= air[:,1:,:]<<7 #up
        exposed[:,1:,:] |= air[:,:-1,:]<<6 #down
        exposed[1:,:,:] |= air[:-1,:,:]<<5 #left
        exposed[:-1,:,:] |= air[1:,:,:]<<4 #right
        exposed[:,:,:-1] |= air[:,:,1:]<<3 #forward
        exposed[:,:,1:] |= air[:,:,:-1]<<2 #back
        self.exposed = exposed*(~air)

    def _calc_vertex_data(self,position):
        self._calc_exposed_faces()
        exposed = self.exposed[1:-1,:,1:-1]
        sh = exposed.shape
        exposed = exposed.swapaxes(0,2).reshape(sh[0]*sh[1]*sh[2])
        egz = exposed>0
        pos = SECTOR_GRID[egz] + position
        exposed = exposed[egz]
        exposed = numpy.unpackbits(exposed[:,numpy.newaxis],axis=1)
        exposed = numpy.array(exposed,dtype=bool)
        exposed = exposed[:,:6]
        #b = self[pos]
        p = (pos - numpy.array(position)).T
        b = self.blocks[1:-1,:,1:-1][p[0],p[1],p[2]]
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

    def get_block(self, position, sector_position):
        pos = position - numpy.array(sector_position)
        if len(pos.shape)>1:
            pos = pos.T
            return self.blocks[1:-1,:,1:-1][pos[0],pos[1],pos[2]]
        return self.blocks[1:-1,:,1:-1][pos[0],pos[1],pos[2]]

    def set_block(self, position, sector_position, val):
        pos = position - numpy.array(sector_position)
        if len(pos.shape)>1:
            pos = pos.T
            self.blocks[1:-1,:,1:-1][pos[0],pos[1],pos[2]] = val
        self.blocks[1:-1,:,1:-1][pos[0],pos[1],pos[2]] = val


class Sector(object):
    def __init__(self,position,model):
        self.position = position[0],0,position[2]
        self.bposition = position[0]-1,0,position[2]-1 #block relative position (the sector is overgenerated by one block along each x,z edge)
        self.model = model
        self.blocks = numpy.zeros((SECTOR_SIZE+2,SECTOR_HEIGHT,SECTOR_SIZE+2),dtype='u2')

        self.vt_data = None
        self.exposed = None

    def __getitem__(self, position):
        pos = position - numpy.array(self.bposition)
        if len(pos.shape)>1:
            pos = pos.T
            return self.blocks[pos[0],pos[1],pos[2]]
        return self.blocks[pos[0],pos[1],pos[2]]

    def __setitem__(self, position, value):
        pos = position - numpy.array(self.bposition)
        if len(pos.shape)>1:
            pos = pos.T
        self.blocks[pos[0],pos[1],pos[2]] = value

    def invalidate(self):
        self.invalidate_vt = True
        self.vt_data = None

    def calc_exposed_faces(self):
        #TODO: The 3D bitwise ops are slow
        air = BLOCK_SOLID[self.blocks] == 0

        light = numpy.cumproduct(air[:,::-1,:], axis=1)[:,::-1,:]

        exposed = numpy.zeros(air.shape,dtype=numpy.uint8)
        exposed[:,:-1,:] |= air[:,1:,:]<<7 #up
        exposed[:,1:,:] |= air[:,:-1,:]<<6 #down
        exposed[1:,:,:] |= air[:-1,:,:]<<5 #left
        exposed[:-1,:,:] |= air[1:,:,:]<<4 #right
        exposed[:,:,:-1] |= air[:,:,1:]<<3 #forward
        exposed[:,:,1:] |= air[:,:,:-1]<<2 #back
        self.exposed = exposed*(~air)

    def calc_vertex_data(self):
        if self.exposed == None:
            self.calc_exposed_faces()
        if self.vt_data == None:
            exposed = self.exposed[1:-1,:,1:-1]
            sh = exposed.shape
            exposed = exposed.swapaxes(0,2).reshape(sh[0]*sh[1]*sh[2])
            egz = exposed>0
            pos = SECTOR_GRID[egz] + self.position
            exposed = exposed[egz]
            exposed = numpy.unpackbits(exposed[:,numpy.newaxis],axis=1)
            exposed = numpy.array(exposed,dtype=bool)
            exposed = exposed[:,:6]
            #b = self[pos]
            p = (pos - numpy.array(self.position)).T
            b = self.blocks[1:-1,:,1:-1][p[0],p[1],p[2]]
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

    def set_block(self, position, block_id):
        """ Set a block in the sector at `position` to `block_id`.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to add.
        texture : list of len 3
            The coordinates of the texture squares. Use `tex_coords()` to
            generate.
        """
        self[position] = block_id
        self.invalidate()
        self.update_block(position)
        self.model.check_neighbors(position)

    def update_block(self, position):
        """ Update the faces of the block at the given `position`.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to show.
        """
        sector_pos = numpy.array(position) - self.bposition
        x, y, z = sector_pos
        exposed = 0
        if self[position] != 0:
            i=1
            for f in FACES:
                if not BLOCK_SOLID[self.model[numpy.array(position)+numpy.array(f)]]:
                    exposed |= 1<<(8-i)
                i+=1
        if exposed != self.exposed[x,y,z]:
            self.exposed[x,y,z] = exposed
            self.invalidate()

    def _initialize(self):
        """ Initialize the sector by procedurally generating terrain using
        simplex noise.

        """
        self.blocks = mapgen.generate_sector(self.position, None, None)


class Model(object):

    def __init__(self):
        mapgen.initialize_map_generator()
        # The world is stored in sector chunks.
        self.sectors = {}
        self.sector_cache = []

        # Simple function queue implementation. The queue is populated with
        # _show_block() and _hide_block() calls
#        self.queue = deque()

        d = range(-SECTOR_SIZE*3,SECTOR_SIZE*3+1,SECTOR_SIZE)
        #d = range(-128,128+1,SECTOR_SIZE)
        for pos in itertools.product(d,(0,),d):
            s=Sector(pos, self)
            self.sectors[sectorize(pos)] = s
            s._initialize()
        for s in self.sectors:
            self.sectors[s].calc_vertex_data()


    def __getitem__(self, position):
        """
        retrieves the block at the (x,y,z) coordinate tuple `position`
        """
        try:
            return self.sectors[sectorize(position)][position]
        except:
            return None

    def neighbor_sectors(self, pos):
        """
        return a tuple (dx, dz, sector) of currently loaded neighbors to the sector at pos
        """
        pos = sectorize(pos)
        for x in ((-1,0),(1,0),(0,-1),(0,1)):
            npos = (pos[0]+x[0]*SECTOR_SIZE,0,pos[2]+x[1]*SECTOR_SIZE)
            if npos in self.sectors:
                yield x[0],x[1],self.sectors[npos]

    def change_sectors(self, old, new):
        """
        the observer has moved from sector old to new
        """
        if self.thread == None:
            self.thread = threading.Thread(target = self._load_sectors, args = (set(self.sectors),new))
            self.thread.start()

    def request_sector(self, spos):
        spos = sectorize(spos)
        if spos in self.sectors:
            if self.sectors[spos].vt_data == None:
                self.sectors[spos].calc_vertex_data()
        else:
            t0=time.time()
            s = Sector(spos,self)
            t1 = time.time()
            s._initialize()
            t2 = time.time()
##Eventually need to turn this back on, but the logic isn't right (should be dropping positions that are "far" from active players)
#            if len(self.sector_cache)>LOADED_SECTORS*LOADED_SECTORS+20:
#                sp = self.sector_cache.pop(0)
#                if (sp[0] - spos[0])**2 + (sp[2] - spos[2])**2 < LOADED_SECTORS*LOADED_SECTORS:
#                    self.sector_cache.append(sp)
#                else:
#                    print('dropping',self.sector_cache[0])
#                    print('***',len(self.sector_cache))
#                    del self.sectors[sp]
            self.sector_cache.append(spos)
            self.sectors[spos] = s
            t3 = time.time()
            s.calc_vertex_data()
            t4 = time.time()
            print('sector creation',spos,'timings',t1-t0,t2-t1,t3-t2,t4-t3)
        return self.sectors[spos].vt_data, self.sectors[spos].blocks

    def set_block(self, position, block):
        position = normalize(position)
        spos = sectorize(position)
        print('set block',spos, position)
        result = []
        result.append(self._set_block(spos, position, block))
        if position[0] - spos[0] == 0:
            nspos = sectorize((position[0]-1, position[1], position[2]))
            result.append(self._set_block(nspos, position, block))
        if position[0] - spos[0] == SECTOR_SIZE-1:
            nspos = sectorize((position[0]+1, position[1], position[2]))
            result.append(self._set_block(nspos, position, block))
        if position[2] - spos[2] == 0:
            nspos = sectorize((position[0], position[1], position[2]-1))
            result.append(self._set_block(nspos, position, block))
        if position[2] - spos[2] == SECTOR_SIZE-1:
            nspos = sectorize((position[0], position[1], position[2]+1))
            result.append(self._set_block(nspos, position, block))
        return result

    def _set_block(self, spos, position, block):
        if spos not in self.sectors:
            ##TODO: This causes a fatal error
            return None
        s = self.sectors[spos]
        s.set_block(position, block)
        s.calc_vertex_data()
        return spos, s.vt_data, s.blocks

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

    def check_neighbors(self, position):
        """ Check all blocks surrounding `position` and ensure their visual
        state is current. This means hiding blocks that are not exposed and
        ensuring that all exposed blocks are shown. Usually used after a block
        is added or removed.

        """
        x, y, z = position
        for dx, dy, dz in FACES:
            key = (x + dx, y + dy, z + dz)
            b = self[key]
            if not BLOCK_SOLID[b] or b is None:
                continue
#            self.update_block(key)
            self.sectors[sectorize(key)].update_block(key)

class Loader(object):
    def __init__(self, server_ip = None):
        print('starting loader at %s:%i'%(LOADER_IP,LOADER_PORT))
        self.listener = multiprocessing.connection.Listener(address = (LOADER_IP,LOADER_PORT), authkey = 'password')
        if server_ip is not None:
            print('Connecting to multiplayer server',server_ip,':',SERVER_PORT)
            self.mp_server = multiprocessing.connection.Client(address = (server_ip, SERVER_PORT), authkey = 'password')
        else:
            self.mp_server = None
        self.connections = []
        self.world = Model()

    def accept_connection(self):
        conn = self.listener.accept()
        self.connections.append(conn)
        return conn

    def serve(self):
        alive = True
        while alive:
            if self.mp_server is not None:
                r,w,x = select.select([self.mp_server, self.listener._listener._socket] + self.connections, [], [])
            else:
                r,w,x = select.select([self.listener._listener._socket] + self.connections, [], [])
            accept_new = True
            for conn in self.connections:
                if conn in r:
                    accept_new = False
                    print('recv from',conn)
                    try:
                        msg, data = conn.recv()
                        print('client call',msg)
                        if msg == 'request_sector':
                            spos = data
                            sector_data, blocks = self.world.request_sector(spos)
                            conn.send_bytes(cPickle.dumps([spos, blocks, sector_data],-1))
                        if msg == 'add_block':
                            pos, block = data
                            if self.mp_server is not None:
                                self.mp_server.send(['add_block',[pos, block]])
                            for spos, sector_data, blocks in self.world.set_block(pos, block):
                                conn.send_bytes(cPickle.dumps([spos, blocks, sector_data],-1))
                        if msg == 'remove_block':
                            pos = data[0]
                            for spos, sector_data, blocks in self.world.set_block(pos, 0):
                                print('sending sector data',spos)
                                conn.send_bytes(cPickle.dumps([spos, blocks, sector_data],-1))
                        if msg == 'quit':
                            alive = False
                    except EOFError:
                        print('Lost',conn)
                        self.connections.remove(conn)
            if self.mp_server is not None and self.mp_server in r:
                msg,data = conn.recv()
                print('msg',data)
            if accept_new:
                conn = self.accept_connection()
                print('accept',conn)
        self.listener.close()

def _loader_main(ip):
    l = Loader(ip)
    l.serve()

def start_loader(ip = None):
    process = multiprocessing.Process(target = _loader_main, args = (ip,))
    process.start()

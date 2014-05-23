# standard library imports
import math
import itertools
import threading
import time
import numpy
import select
import cPickle
import multiprocessing.connection
import socket
import sys

def get_network_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.connect(('<broadcast>', 0))
    return s.getsockname()[0]

# local imports
from config import SECTOR_SIZE, SECTOR_HEIGHT, LOADED_SECTORS, SERVER_IP, SERVER_PORT
from util import normalize, sectorize, FACES, cube_v, cube_v2
from blocks import BLOCK_VERTICES, BLOCK_COLORS, BLOCK_NORMALS, BLOCK_TEXTURES, BLOCK_ID, BLOCK_SOLID, TEXTURE_PATH
import noise

SECTOR_GRID = numpy.mgrid[:SECTOR_SIZE,:SECTOR_HEIGHT,:SECTOR_SIZE].T
SH = SECTOR_GRID.shape
SECTOR_GRID = SECTOR_GRID.reshape((SH[0]*SH[1]*SH[2],3))

t = int(time.time())
noisen = noise.SimplexNoise(seed=t)
noisen1 = noise.SimplexNoise(seed=t+342)
noisen2 = noise.SimplexNoise(seed=t+434345)


class Sector(object):
    def __init__(self,position,model):
        self.position = position[0],-40,position[2]
        self.model = model
        self.blocks = numpy.zeros((SECTOR_SIZE,SECTOR_HEIGHT,SECTOR_SIZE),dtype='u2')

        self.vt_data = None
        self.exposed = None

    def __getitem__(self, position):
#        position = normalize(position)
        pos = position - numpy.array(self.position)
        if len(pos.shape)>1:
            pos = pos.T
            return self.blocks[pos[0],pos[1],pos[2]]
        return self.blocks[pos[0],pos[1],pos[2]]

    def __setitem__(self, position, value):
        position = normalize(position)
        pos = position - numpy.array(self.position)
        if len(pos.shape)>1:
            pos = pos.T
        self.blocks[pos[0],pos[1],pos[2]] = value

    def update_edge(self, dx, dz, sector):
        if self.exposed is None:
            return
        if dx>0:
            b0 = (self.blocks[-1,:,:]!=0)&(BLOCK_SOLID[sector.blocks[0,:,:]]==0)
            self.exposed[-1,:,:] |= b0 << 4 #right edge
            self.invalidate()
        elif dx<0:
            b0 = (self.blocks[0,:,:]!=0)&(BLOCK_SOLID[sector.blocks[-1,:,:]]==0)
            self.exposed[0,:,:] |= b0 << 5 #left edge
            self.invalidate()
        elif dz>0:
            b0 = (self.blocks[:,:,-1]!=0)&(BLOCK_SOLID[sector.blocks[:,:,0]]==0)
            self.exposed[:,:,-1] |= b0 << 3 #front edge
            self.invalidate()
        elif dz<0:
            b0 = (self.blocks[:,:,0]!=0)&(BLOCK_SOLID[sector.blocks[:,:,-1]]==0)
            self.exposed[:,:,0] |= b0 << 2 #back edge
            self.invalidate()

    def invalidate(self):
        self.invalidate_vt = True
        self.vt_data = None

    def edge_blocks(self,dx=0,dz=0):
        pos = self.position
        try:
            s=self.model.sectors[sectorize((pos[0]+dx*SECTOR_SIZE,pos[1],pos[2]+dz*SECTOR_SIZE))]
        except KeyError:
            s=None
        if s is not None:
            if dx>0:
                return BLOCK_SOLID[s.blocks[0,:,:]]==0
            elif dx<0:
                return BLOCK_SOLID[s.blocks[-1,:,:]]==0
            elif dz>0:
                return BLOCK_SOLID[s.blocks[:,:,0]]==0
            elif dz<0:
                return BLOCK_SOLID[s.blocks[:,:,-1]]==0
        else:
            if dx>0:
                return numpy.zeros((SECTOR_HEIGHT,SECTOR_SIZE),dtype=bool)
            elif dx<0:
                return numpy.zeros((SECTOR_HEIGHT,SECTOR_SIZE),dtype=bool)
            elif dz>0:
                return numpy.zeros((SECTOR_SIZE,SECTOR_HEIGHT),dtype=bool)
            elif dz<0:
                return numpy.zeros((SECTOR_SIZE,SECTOR_HEIGHT),dtype=bool)

    def calc_exposed_faces(self):
        #TODO: The 3D bitwise ops are slow
        t = time.time()
        air = BLOCK_SOLID[self.blocks] == 0

        light = numpy.cumproduct(air[:,::-1,:], axis=1)[:,::-1,:]

        exposed = numpy.zeros(air.shape,dtype=numpy.uint8)
        exposed[0,:,:] |= self.edge_blocks(dx=-1)<<5 #left edge
        exposed[-1,:,:] |= self.edge_blocks(dx=1)<<4 #right edge
        exposed[:,:,0] |= self.edge_blocks(dz=-1)<<2 #back edge
        exposed[:,:,-1] |= self.edge_blocks(dz=1)<<3 #front edge
        exposed[:,:-1,:] |= air[:,1:,:]<<7 #up
        exposed[:,1:,:] |= air[:,:-1,:]<<6 #down
        exposed[1:,:,:] |= air[:-1,:,:]<<5 #left
        exposed[:-1,:,:] |= air[1:,:,:]<<4 #right
        exposed[:,:,:-1] |= air[:,:,1:]<<3 #forward
        exposed[:,:,1:] |= air[:,:,:-1]<<2 #back
        self.exposed = exposed*(~air)

    def check_show(self):
        if self.exposed == None:
            self.calc_exposed_faces()
        if self.vt_data == None:
            sh = self.exposed.shape
            exposed = self.exposed.swapaxes(0,2).reshape(sh[0]*sh[1]*sh[2])
            egz = exposed>0
            pos = SECTOR_GRID[egz] + self.position
            exposed = exposed[egz]
            exposed = numpy.unpackbits(exposed[:,numpy.newaxis],axis=1)
            exposed = numpy.array(exposed,dtype=bool)
            exposed = exposed[:,:6]
            b = self[pos]
            texture_data = BLOCK_TEXTURES[b]
            color_data = BLOCK_COLORS[b]
#            color_data = BLOCK_COLORS[b]*(0.25+0.75*exposed_light[:,:,numpy.newaxis])
            normal_data = numpy.tile(BLOCK_NORMALS,(len(b),1,4))#*exposed_light[:,:,numpy.newaxis]
            vertex_data = 0.5*BLOCK_VERTICES[b] + numpy.tile(pos,4)[:,numpy.newaxis,:]

            v = vertex_data[exposed].ravel()
            t = texture_data[exposed].ravel()
            n = normal_data[exposed].ravel()
            c = color_data[exposed].ravel()

            count = len(v)/3
            self.vt_data = (count, v, t, n, c)

    def add_block(self, position, texture):
        """ Add a block with the given `texture` and `position` to the world.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to add.
        texture : list of len 3
            The coordinates of the texture squares. Use `tex_coords()` to
            generate.
        """
        position = normalize(position)
        if self[position] != 0:
            self.remove_block(position, immediate)
        self[position] = texture
        self.update_block(position)
        self.model.check_neighbors(position)

    def remove_block(self, position):
        """ Remove the block at the given `position`.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to remove.
        """
        self[position] = 0
        self.update_block(position)
        self.model.check_neighbors(position)

    def update_block(self, position):
        """ Update the faces of the block at the given `position`.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to show.
        """
        sector_pos = numpy.array(position) - self.position
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
        STONE = BLOCK_ID['Stone']
        GRASS = BLOCK_ID['Grass']

        N = SECTOR_SIZE
        STEP = 40.0
        Z = numpy.mgrid[0:N,0:N].T/STEP
        shape = Z.shape
        Z = Z.reshape((shape[0]*shape[1],2))
        Z = Z + numpy.array([self.position[0],self.position[2]])/STEP

        N1=noisen.noise(Z)*30
        N2=noisen1.noise(Z)*30-20
        N3=noisen2.noise(Z)*30-30
        #N1 = ((N1 - N1.min())/(N1.max() - N1.min()))*20
        N1 = N1.reshape((SECTOR_SIZE,SECTOR_SIZE))
        N2 = N2.reshape((SECTOR_SIZE,SECTOR_SIZE))
        N3 = N3.reshape((SECTOR_SIZE,SECTOR_SIZE))

        #N2 = (N2 - N2.min())/(N2.max() - N2.min())*30
        Z = Z*STEP + numpy.array([self.position[0],self.position[2]])
        b = numpy.zeros((SECTOR_HEIGHT,SECTOR_SIZE,SECTOR_SIZE),dtype='u2')
        for y in range(SECTOR_HEIGHT):
            b[y] = ((y-40<N1-2)*STONE + (((y-40>=N1-2) & (y-40<N1))*GRASS))*(1 - (y-40>N3)*(y-40<N2)*(y>10))
        self.blocks = b.swapaxes(0,1).swapaxes(0,2)


class Model(object):

    def __init__(self):

        # The world is stored in sector chunks.
        self.sectors = {}
        self.sector_lock = threading.Lock()
        self.thread = None

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
            self.sectors[s].check_show()


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
                self.sectors[spos].check_show()
        else:
            t0=time.time()
            s = Sector(spos,self)
            t1 = time.time()
            s._initialize()
            t2 = time.time()
            self.sectors[spos] = s
            t3 = time.time()
            s.check_show()
            t4 = time.time()
            for dx,dz,ns in self.neighbor_sectors(spos):
                ns.update_edge(-dx,-dz,s)
            t5 = time.time()
            print 'sector creation',spos,'timings',t1-t0,t2-t1,t3-t2,t4-t3,t5-t4
        return self.sectors[spos].vt_data, self.sectors[spos].blocks

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

class Server(object):
    def __init__(self):
        print('starting server at %s:%i'%(SERVER_IP,SERVER_PORT))
        self.listener = multiprocessing.connection.Listener(address = (SERVER_IP,SERVER_PORT), authkey = 'password')
        self.connections = []
        self.world = Model()

    def accept_connection(self):
        conn = self.listener.accept()
        self.connections.append(conn)
        return conn

    def serve(self):
        alive = True
        while alive:
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
                            print('sending sector data',spos)
                            conn.send_bytes(cPickle.dumps([spos, sector_data, blocks],-1))
                            #conn.send([spos, sector_data, blocks])
                        if msg == 'kill':
                            alive = False
                    except EOFError:
                        print('Lost',conn)
                        self.connections.remove(conn)
            if accept_new:
                conn = self.accept_connection()
                print('accept',conn)
        self.listener.close()

if __name__ == '__main__':
    if len(sys.argv)>1:
        if sys.argv[1] == 'LAN':
            SERVER_IP = get_network_ip()
    #TODO: use a config file for server settings
    #TODO: use argparse module to override default server settings
    s = Server()
    s.serve()

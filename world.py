# standard library imports
import math
import itertools
import threading
import time
import numpy

# pyglet imports
import pyglet
image = pyglet.image
from pyglet.graphics import TextureGroup
import pyglet.gl as gl

# local imports
from config import SECTOR_SIZE, SECTOR_HEIGHT, LOADED_SECTORS
from util import normalize, sectorize, FACES, cube_v, cube_v2
from blocks import BLOCK_VERTICES, BLOCK_COLORS, BLOCK_NORMALS, BLOCK_TEXTURES, BLOCK_ID, BLOCK_SOLID, TEXTURE_PATH
import mapgen

SECTOR_GRID = numpy.mgrid[:SECTOR_SIZE,:SECTOR_HEIGHT,:SECTOR_SIZE].T
SH = SECTOR_GRID.shape
SECTOR_GRID = SECTOR_GRID.reshape((SH[0]*SH[1]*SH[2],3))


class Sector(object):
    def __init__(self,position,group,model,shown=True):
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
        self.exposed = None

    def draw(self):
        if self.invalidate_vt:
            self.check_show()
            self.invalidate_vt = False
        self.batch.draw()

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
        
    def grid(self):
        grid = SECTOR_GRID + numpy.array(self.position)
        return grid

    def __iter__(self):
        grid = numpy.mgrid[:SECTOR_SIZE,:SECTOR_HEIGHT,:SECTOR_SIZE].T
        sh = grid.shape
        grid = grid.reshape((sh[0]*sh[1]*sh[2],3))
        grid += numpy.array(self.position)
        #grid = grid[self.blocks.reshape((sh[0]*sh[1]*sh[2]))>0]
        for m in grid:
            yield m[0],m[1],m[2]
    
    def add_block(self, position, texture, immediate=True):
        """ Add a block with the given `texture` and `position` to the world.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to add.
        texture : list of len 3
            The coordinates of the texture squares. Use `tex_coords()` to
            generate.
        immediate : bool
            Whether or not to draw the block immediately.

        """                
        position = normalize(position)
        if self[position] != 0:
            self.remove_block(position, immediate)
        self[position] = texture
        self.update_block(position)
        self.model.check_neighbors(position)
    
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
                
    def check_show(self,add_to_batch = True):
        if self.exposed == None:
            self.calc_exposed_faces()
        if self.vt is not None:
            self.vt.delete()
            self.vt=None
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

        if add_to_batch:
            (count, v, t, n, c) = self.vt_data
            self.vt = self.batch.add(count, gl.GL_QUADS, self.group,
                ('v3f/static', v),
                ('t2f/static', t),
                ('n3f/static', n),
                ('c3B/static', c))
            self.vt_data = None

    def remove_block(self, position, immediate=True):
        """ Remove the block at the given `position`.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to remove.
        immediate : bool
            Whether or not to immediately remove block from canvas.

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
        self.blocks = mapgen.generate_sector(self.position, self, self.model)


class Model(object):

    def __init__(self):

        # A TextureGroup manages an OpenGL texture.
        self.group = TextureGroup(image.load(TEXTURE_PATH).get_texture())
        
        mapgen.initialize_map_generator()

        # The world is stored in sector chunks.
        self.sectors = {}
        self.sector_lock = threading.Lock()
        self.thread = None

        # Simple function queue implementation. The queue is populated with
        # _show_block() and _hide_block() calls
#        self.queue = deque()

        d = range(-SECTOR_SIZE,SECTOR_SIZE+1,SECTOR_SIZE)
        #d = range(-128,128+1,SECTOR_SIZE)
        for pos in itertools.product(d,(0,),d):
            s=Sector(pos, self.group, self)
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

    def draw(self, position, (center, radius)):
        #t = time.time()
        with self.sector_lock:
            for s in self.sectors:
                spos = numpy.array([s[0], s[2]])+SECTOR_SIZE/2
                if ((center-spos)**2).sum()>(radius+SECTOR_SIZE/2)**2:
                    continue
                if self.sectors[s].shown:
                    self.sectors[s].draw()
        #print 'draw',time.time() -t 
        
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

    def _load_sectors(self, old_sectors_pos, new_pos):
        #TODO: Need to update the edges of blocks that are at the edge of the map
        dt = 1e-4
    
        adds = 0
        G = range(-LOADED_SECTORS,LOADED_SECTORS+1)
        sectors_pos = set()
        for dx,dy,dz in itertools.product(G,(0,),G):
            pos = numpy.array([new_pos[0],new_pos[1],new_pos[2]]) \
                + numpy.array([dx*SECTOR_SIZE,dy,dz*SECTOR_SIZE])
            pos = sectorize(pos)
            sectors_pos.add(pos)
            if pos not in old_sectors_pos:
                t = time.time()
                adds += 1
                s = Sector(pos,self.group,self,False)
                s._initialize()
                time.sleep(dt)
                s.calc_exposed_faces()
                with self.sector_lock:
                    self.sectors[pos] = s
                s.invalidate()
                s.check_show(add_to_batch = False)
                s.shown = True
                with self.sector_lock:
                    for dx,dz,ns in self.neighbor_sectors(pos):
                        ns.update_edge(-dx,-dz,s)
                time.sleep(dt)
                print 'sector',pos,'took',time.time()-t
        #TODO: should probably lock the mutex for this
        removes = 0
        with self.sector_lock:
            for pos in old_sectors_pos:
                if pos not in sectors_pos:
                    del self.sectors[pos]
                    removes += 1
        self.thread=None
        print 'added',adds
        print 'removed',removes

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


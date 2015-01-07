import math
import random
import time
import sys

# pyglet imports
import pyglet
image = pyglet.image
from pyglet.graphics import TextureGroup
from pyglet.window import key, mouse
import pyglet.gl as gl

GLfloat3 = gl.GLfloat*3
GLfloat4 = gl.GLfloat*4

# standard lib imports
from collections import deque
import numpy
import itertools

# local module imports
import world_proxy as world
import util
import config
from config import DIST, TICKS_PER_SEC, FLYING_SPEED, GRAVITY, JUMP_SPEED, \
        MAX_JUMP_HEIGHT, PLAYER_HEIGHT, TERMINAL_VELOCITY, TICKS_PER_SEC, \
        WALKING_SPEED
from blocks import BLOCK_ID, BLOCK_TEXTURES, BLOCK_VERTICES, BLOCK_COLORS


class Window(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)


        # Whether or not the window exclusively captures the mouse.
        self.exclusive = False

        # When flying gravity has no effect and speed is increased.
        self.flying = True
        self.fly_climb = 0

        # Strafing is moving lateral to the direction you are facing,
        # e.g. moving to the left or right while continuing to face forward.
        #
        # First element is -1 when moving forward, 1 when moving back, and 0
        # otherwise. The second element is -1 when moving left, 1 when moving
        # right, and 0 otherwise.
        self.strafe = [0, 0]

        # Current (x, y, z) position in the world, specified with floats. Note
        # that, perhaps unlike in math class, the y-axis is the vertical axis.
        self.position = (0, 160, 0)

        # First element is rotation of the player in the x-z plane (ground
        # plane) measured from the z-axis down. The second is the rotation
        # angle from the ground plane up. Rotation is in degrees.
        #
        # The vertical plane rotation ranges from -90 (looking straight down) to
        # 90 (looking straight up). The horizontal rotation range is unbounded.
        self.rotation = (0, 0)

        # Which sector the player is currently in.
        self.sector = None

        # The crosshairs at the center of the screen.
        self.reticle = None

        self.inventory_batch = pyglet.graphics.Batch()
        self.inventory_item = None

        # Velocity in the y (upward) direction.
        self.dy = 0

        # A list of blocks the player can place. Hit num keys to cycle.
        self.inventory = list(BLOCK_ID)

        # The current block the user can place. Hit num keys to cycle.
        self.block = self.inventory[0]

        # Convenience list of num keys.
        self.num_keys = [
            key._1, key._2, key._3, key._4, key._5,
            key._6, key._7, key._8, key._9, key._0]

        # Instance of the model that handles the world.
        self.model = world.ModelProxy()

        self.inventory_group = util.InventoryGroup(parent = self.model.group)
#        line_group = util.LineDrawGroup(thickness = 3)
#        self.inventory_outline_group = util.InventoryOutlineGroup(parent = line_group)

        # The label that is displayed in the top left of the canvas.
        self.label = pyglet.text.Label('', font_name='Arial', font_size=18,
            x=10, y=self.height - 10, anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))

        # This call schedules the `update()` method to be called
        # TICKS_PER_SEC. This is the main game event loop.
        pyglet.clock.schedule_interval(self.update, 1.0 / TICKS_PER_SEC)

    def set_exclusive_mouse(self, exclusive):
        """ If `exclusive` is True, the game will capture the mouse, if False
        the game will ignore the mouse.

        """
        super(Window, self).set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    def get_sight_vector(self):
        """ Returns the current line of sight vector indicating the direction
        the player is looking.

        """
        x, y = self.rotation
        # y ranges from -90 to 90, or -pi/2 to pi/2, so m ranges from 0 to 1 and
        # is 1 when looking ahead parallel to the ground and 0 when looking
        # straight up or down.
        m = math.cos(math.radians(y))
        # dy ranges from -1 to 1 and is -1 when looking straight down and 1 when
        # looking straight up.
        dy = math.sin(math.radians(y))
        dx = math.cos(math.radians(x - 90)) * m
        dz = math.sin(math.radians(x - 90)) * m
        return (dx, dy, dz)

    def get_motion_vector(self):
        """ Returns the current motion vector indicating the velocity of the
        player.

        Returns
        -------
        vector : tuple of len 3
            Tuple containing the velocity in x, y, and z respectively.

        """
        if any(self.strafe):
            x, y = self.rotation
            strafe = math.degrees(math.atan2(*self.strafe))
            y_angle = math.radians(y)*any(self.strafe)
            x_angle = math.radians(x + strafe)*any(self.strafe)
            if self.flying:
                m = math.cos(y_angle)
                dy = math.sin(y_angle)
                if self.strafe[1]:
                    # Moving left or right.
                    dy = 0.0
                    m = 1
                if self.strafe[0] > 0:
                    # Moving backwards.
                    dy *= -1
                # When you are flying up or down, you have less left and right
                # motion.
                dx = math.cos(x_angle) * m
                dz = math.sin(x_angle) * m
            else:
                dy = 0.0
                dx = math.cos(x_angle)
                dz = math.sin(x_angle)
        else:
            dy = 0.0
            dx = 0.0
            dz = 0.0
        if self.flying and self.fly_climb!=0:
            dy = self.fly_climb
        return (dx, dy, dz)

    def update(self, dt):
        """ This method is scheduled to be called repeatedly by the pyglet
        clock.

        Parameters
        ----------
        dt : float
            The change in time since the last call.

        """
#        self.model.process_queue()
        sector = util.sectorize(self.position)
        self.model.update_sectors(self.sector, sector)
        self.sector = sector
        m = 20
        dt = min(dt, 0.2)
        for _ in range(m):
            self._update(dt / m)

    def _update(self, dt):
        """ Private implementation of the `update()` method. This is where most
        of the motion logic lives, along with gravity and collision detection.

        Parameters
        ----------
        dt : float
            The change in time since the last call.

        """
        # walking
        speed = FLYING_SPEED if self.flying else WALKING_SPEED
        d = dt * speed # distance covered this tick.
        dx, dy, dz = self.get_motion_vector()
        # New position in space, before accounting for gravity.
        dx, dy, dz = dx * d, dy * d, dz * d
        # gravity
        if not self.flying:
            # Update your vertical speed: if you are falling, speed up until you
            # hit terminal velocity; if you are jumping, slow down until you
            # start falling.
            self.dy -= dt * GRAVITY
            self.dy = max(self.dy, -TERMINAL_VELOCITY)
            dy += self.dy * dt
        # collisions
        x, y, z = self.position
        x, y, z = self.collide((x + dx, y + dy, z + dz), PLAYER_HEIGHT)
        self.position = (x, y, z)

    def collide(self, position, height):
        """ Checks to see if the player at the given `position` and `height`
        is colliding with any blocks in the world.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position to check for collisions at.
        height : int or float
            The height of the player.

        Returns
        -------
        position : tuple of len 3
            The new position of the player taking into account collisions.

        """
        # How much overlap with a dimension of a surrounding block you need to
        # have to count as a collision. If 0, touching terrain at all counts as
        # a collision. If .49, you sink into the ground, as if walking through
        # tall grass. If >= .5, you'll fall through the ground.
        pad = 0.1
        p = list(position)
        np = util.normalize(position)
        for face in util.FACES:  # check all surrounding blocks
            for i in range(3):  # check each dimension independently
                if not face[i]:
                    continue
                # How much overlap you have with this dimension.
                d = (p[i] - np[i]) * face[i]
                if d < pad:
                    continue
                for dy in range(height):  # check each height
                    op = list(np)
                    op[1] -= dy
                    op[i] += face[i]
                    b = self.model[util.normalize(op)]
                    if b==0 or b is None:
                        continue
                    p[i] -= (d - pad) * face[i]
                    if face == (0, -1, 0) or face == (0, 1, 0):
                        # You are colliding with the ground or ceiling, so stop
                        # falling / rising.
                        self.dy = 0
                    break
        return tuple(p)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        ind = self.inventory.index(self.block)
        ind += scroll_y
        if ind >= len(self.inventory):
            ind -= len(self.inventory)
        if ind < 0:
            ind += len(self.inventory)
        self.block = self.inventory[ind]
        self.update_inventory_item_batch()

    def on_mouse_press(self, x, y, button, modifiers):
        """ Called when a mouse button is pressed. See pyglet docs for button
        amd modifier mappings.

        Parameters
        ----------
        x, y : int
            The coordinates of the mouse click. Always center of the screen if
            the mouse is captured.
        button : int
            Number representing mouse button that was clicked. 1 = left button,
            4 = right button.
        modifiers : int
            Number representing any modifying keys that were pressed when the
            mouse button was clicked.

        """
        if self.exclusive:
            vector = self.get_sight_vector()
            block, previous = self.model.hit_test(self.position, vector)
            if (button == mouse.RIGHT) or \
                    ((button == mouse.LEFT) and (modifiers & key.MOD_CTRL)):
                # ON OSX, control + left click = right click.
                if previous:
                    px, py, pz = util.normalize(self.position)
                    if not (previous == (px, py, pz) or previous == (px, py-1, pz)):
                        self.model.add_block(previous, BLOCK_ID[self.block])
            elif button == pyglet.window.mouse.LEFT and block:
                self.model.remove_block(block)
        else:
            self.set_exclusive_mouse(True)

    def on_mouse_motion(self, x, y, dx, dy):
        """ Called when the player moves the mouse.

        Parameters
        ----------
        x, y : int
            The coordinates of the mouse click. Always center of the screen if
            the mouse is captured.
        dx, dy : float
            The movement of the mouse.

        """
        if self.exclusive:
            m = 0.15
            x, y = self.rotation
            x, y = x + dx * m, y + dy * m
            y = max(-90, min(90, y))
            self.rotation = (x, y)

    def on_key_press(self, symbol, modifiers):
        """ Called when the player presses a key. See pyglet docs for key
        mappings.

        Parameters
        ----------
        symbol : int
            Number representing the key that was pressed.
        modifiers : int
            Number representing any modifying keys that were pressed.

        """
        if symbol == key.W:
            self.strafe[0] -= 1
        elif symbol == key.S:
            self.strafe[0] += 1
        elif symbol == key.A:
            self.strafe[1] -= 1
        elif symbol == key.D:
            self.strafe[1] += 1
        elif symbol == key.SPACE:
            if self.flying:
                self.fly_climb += 1
            if self.dy == 0:
                self.dy = JUMP_SPEED
        elif symbol == key.LSHIFT:
            if self.flying:
                self.fly_climb -= 1
        elif symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif symbol == key.TAB:
            self.flying = not self.flying
        elif symbol in self.num_keys:
            index = (symbol - self.num_keys[0]) % len(self.inventory)
            self.block = self.inventory[index]
            self.update_inventory_item_batch()

    def on_key_release(self, symbol, modifiers):
        """ Called when the player releases a key. See pyglet docs for key
        mappings.

        Parameters
        ----------
        symbol : int
            Number representing the key that was pressed.
        modifiers : int
            Number representing any modifying keys that were pressed.

        """
        if symbol == key.W:
            self.strafe[0] += 1
        elif symbol == key.S:
            self.strafe[0] -= 1
        elif symbol == key.A:
            self.strafe[1] += 1
        elif symbol == key.D:
            self.strafe[1] -= 1
        elif symbol == key.SPACE:
            self.fly_climb = 0
        elif symbol == key.LSHIFT:
            self.fly_climb = 0

    def on_resize(self, width, height):
        """ Called when the window is resized to a new `width` and `height`.

        """
        # label
        self.label.y = height - 10
        # reticle
        if self.reticle:
            self.reticle.delete()
        x, y = self.width / 2, self.height / 2
        n = 10
        self.reticle = pyglet.graphics.vertex_list(4,
            ('v2i', (x - n, y, x + n, y, x, y - n, x, y + n))
        )
        #inventory item
        self.update_inventory_item_batch()

    def update_inventory_item_batch(self):
        size = 64
        if self.inventory_item is not None:
            self.inventory_item.delete()
#            self.inventory_item_outline.delete()
        #texture cube
        t = BLOCK_TEXTURES[BLOCK_ID[self.block]][:6].ravel()
        v = size/2+size/2*BLOCK_VERTICES[BLOCK_ID[self.block]] + numpy.tile(numpy.array([16,16+size/2,0]),4)
        v = v.ravel()
        c = BLOCK_COLORS[BLOCK_ID[self.block]][:6].ravel()
        self.inventory_item = self.inventory_batch.add(len(t)/2, gl.GL_QUADS, self.inventory_group,
            ('v3f/static', v),
            ('t2f/static', t),
            ('c3B/static', c),
        )
        #outline
#        v = size/2+(size/2+0.1)*BLOCK_VERTICES[BLOCK_ID[self.block]] + numpy.tile(numpy.array([16,16+size/2,0]),4)
#        v = numpy.hstack((v[:,:3],v,v[:,-3:]))
#        v = v.ravel()
#        c = 1*numpy.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1]).repeat(6)
#        self.inventory_item_outline = self.inventory_batch.add(len(v)/3, gl.GL_LINE_STRIP, self.inventory_outline_group,
#            ('v3f/static', v),
#            ('c3B/static', c),
#        )

    def on_close(self):
        self.model.quit()
        pyglet.window.Window.on_close(self)

    def set_2d(self):
        """ Configure OpenGL to draw in 2d.

        """
        width, height = self.get_size()
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -200, 200)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def set_3d(self):
        """ Configure OpenGL to draw in 3d.

        """
        width, height = self.get_size()

        gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(65.0, width / float(height), 0.1, DIST)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        x, y = self.rotation
        gl.glRotatef(x, 0, 1, 0)
        gl.glRotatef(-y, math.cos(math.radians(x)), 0, math.sin(math.radians(x)))
        x, y, z = self.position
        gl.glTranslatef(-x, -y, -z)

        gl.glEnable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHT1)
#        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, GLfloat4(0.05,0.05,0.05,1.0))
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
#        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT)
        #gl.glLightfv(gl.GL_LIGHT1,gl.GL_SPOT_DIRECTION, GLfloat3(0,0,-1))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_AMBIENT, GLfloat4(0.5,0.5,0.5,1.0))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, GLfloat4(1.0,1.0,1.0,1.0))
##        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, GLfloat4(0.35,1.0,0.65,0.0))
        #gl.glLightfv(gl.GL_LIGHT0,gl.GL_SPECULAR, GLfloat4(1,1,1,1))


    def get_frustum_circle(self):
        x,y = self.rotation
        dx = math.cos(math.radians(x - 90))
        dz = math.sin(math.radians(x - 90))

        c = [0,2]
        vec = numpy.array([dx,dz])
        ovec = numpy.array([-dz,dx])
        pos = numpy.array([x for x in self.position])[c]
        center = pos + vec*DIST/2
        far_corner = pos + vec*DIST + ovec*DIST*numpy.tan(65.0/180.0 * numpy.pi)/2
        rad = ((center-far_corner)**2).sum()**0.5/2
        return center,rad

    def on_draw(self):
        """ Called by pyglet to draw the canvas.

        """
        self.clear()
        self.set_3d()
        gl.glColor3d(1, 1, 1)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(0,1)

#        gl.glDepthRange(0.1,1.0)
        self.model.draw(self.position,self.get_frustum_circle())
#        gl.glDepthRange(0.0,0.9)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        self.draw_focused_block()
#        gl.glDepthRange(0.0,1.0)
        self.set_2d()
        self.draw_label()
        self.draw_reticle()
        self.draw_inventory_item()

    def draw_focused_block(self):
        """ Draw black edges around the block that is currently under the
        crosshairs.

        """
        vector = self.get_sight_vector()
        block = self.model.hit_test(self.position, vector)[0]
        if block:
            x, y, z = block
            vertex_data=[]
            for v in util.cube_v([x, y, z], 0.51):
                vertex_data.extend(v)
            gl.glLineWidth(3)
            gl.glColor3d(0, 0, 0)
            #gl.glDepthMask(gl.GL_FALSE)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            #gl.glPolygonOffset(-.1,-.1)
            pyglet.graphics.draw(24, gl.GL_QUADS, ('v3f/static', vertex_data))
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            #gl.glDepthMask(gl.GL_TRUE)
            gl.glLineWidth(1)

    def draw_label(self):
        """ Draw the label in the top left of the screen.

        """
        x, y, z = self.position
        self.label.text = '%02d (%.2f, %.2f, %.2f) %d / %d' % (
            pyglet.clock.get_fps(), x, y, z,
            0,0)#len(self.model._shown), len(self.model.world))
        self.label.draw()

    def draw_reticle(self):
        """ Draw the crosshairs in the center of the screen.

        """
        gl.glColor3d(0, 0, 0)
        self.reticle.draw(gl.GL_LINES)

    def draw_inventory_item(self):
        self.inventory_batch.draw()



def setup_fog():
    """ Configure the OpenGL fog properties.

    """
    # Enable fog. Fog "blends a fog color with each rasterized pixel fragment's
    # post-texturing color."
    gl.glEnable(gl.GL_FOG)
    # Set the fog color.
    gl.glFogfv(gl.GL_FOG_COLOR, (gl.GLfloat * 4)(0.5, 0.69, 1.0, 1))
    # Say we have no preference between rendering speed and quality.
    gl.glHint(gl.GL_FOG_HINT, gl.GL_DONT_CARE)
    # Specify the equation used to compute the blending factor.
    gl.glFogi(gl.GL_FOG_MODE, gl.GL_LINEAR)
    # How close and far away fog starts and ends. The closer the start and end,
    # the denser the fog in the fog range.
    gl.glFogf(gl.GL_FOG_START, 0.75*DIST)
    gl.glFogf(gl.GL_FOG_END, DIST)



def setup():
    """ Basic OpenGL configuration.

    """
    # Set the color of "clear", i.e. the sky, in rgba.
    gl.glClearColor(0.5, 0.69, 1.0, 1)
    # Enable culling (not rendering) of back-facing facets -- facets that aren't
    # visible to you.
    gl.glEnable(gl.GL_CULL_FACE)

    #gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_DST_ALPHA)
    #gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
#    gl.glBlendFunc(gl.GL_ZERO, gl.GL_SRC_COLOR)
#    gl.glEnable(gl.GL_BLEND)
    gl.glAlphaFunc(gl.GL_GREATER, 0.5);
    gl.glEnable(gl.GL_ALPHA_TEST);
    # Set the texture minification/magnification function to GL_NEAREST (nearest
    # in Manhattan distance) to the specified texture coordinates. GL_NEAREST
    # "is generally faster than GL_LINEAR, but it can produce textured images
    # with sharper edges because the transition between texture elements is not
    # as smooth."
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)
    setup_fog()


def main():
    if len(sys.argv)>1:
        config.SERVER_IP = sys.argv[1]
        print('Using server IP address',config.SERVER_IP,':',config.SERVER_PORT)
    window = Window(width=300, height=200, caption='Pyglet', resizable=True)
    # Hide the mouse cursor and prevent the mouse from leaving the window.
    window.set_exclusive_mouse(False)
    setup()
    try:
        pyglet.app.run()
    except:
        import traceback
        traceback.print_exc()
        print('terminating child processes')
        window.model.quit()
        window.set_exclusive_mouse(False)


if __name__ == '__main__':
    main()

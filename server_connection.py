import cPickle
import multiprocessing
import msocket
import select
import traceback

from config import SERVER_PORT
from players import Player
import world_loader


class ClientServerConnectionHandler(object):
    '''
    Handles the low level connection handling details of the multiplayer server
    '''
    def __init__(self, controller_pipe, SERVER_IP):
        print('connecting to server at %s:%i'%(SERVER_IP,SERVER_PORT))
#        self._conn = multiprocessing.connection.Client(address = (SERVER_IP,SERVER_PORT), authkey = 'password')
        self._conn = msocket.Client(SERVER_IP,SERVER_PORT)
        self._pipe = controller_pipe
        self._server_message_queue = []
        self._client_message_queue = []
        self._players = []
        self._fn_dict = {}

    def register_function(self, name, fn):
        self._fn_dict[name]=fn

    def call_function(self, name, *args):
        return self._fn_dict[name](*args)

    def player_from_id(self, id):
        for p in self._players:
            if id == p.id:
                return p

    def communicate_once(self):
        pass

    def communicate_loop(self):
        alive = True
        while alive:
            w = []
            try:
                if len(self._server_message_queue)>0 or self._conn.unfinished_send()>0:
                    w.append(self._conn)
            except AttributeError: #multiprocessing version is blocking
                if len(self._server_message_queue)>0:
                    w.append(self._conn)
            if len(self._client_message_queue)>0:
                w.append(self._pipe)
            r,w,x = select.select([self._conn, self._pipe], w, [])
#            print('select',r,w)
            if self._conn in r:
                try:
                    result = self._conn.recv()
                    if result is not None:
                        msg, pid, data = result
                        print('msg from server',msg)
                        if msg == 'connected':
                            self.connected(pid, *data)
                        elif msg == 'other_player_join':
                            self.other_player_join(pid, *data)
                        else:
                            try:
                                p = self.player_from_id(pid)
                                self.call_function(msg, p, *data)
                            except Exception as ex:
                                traceback.print_exc()
                except EOFError:
                    ##TODO: disconnect from server / tell parent / try to reconnect
                    alive = False
            if self._pipe in r:
                msg, data = self._pipe.recv()
                print('msg from client',msg)
                if msg == 'quit':
                    ##TODO: disconnect from server
                    alive = False
                self._server_message_queue.append((msg, data))
            if self._conn in w:
                self.dispatch_top_server_message()
            if self._pipe in w:
                self.dispatch_top_client_message()
        self._conn.close()
        self._pipe.close()

    def connected(self, player_id, player, players):
        '''
        received when the `player` has successfully joined the game
        '''
        self._players = players
        print('connected', player_id, players)
        for p in players:
            if p.id == player_id:
                self.player = p
                self.send_client('connected', p, self._players)
                return

    def other_player_join(self, player_id, player):
        '''
        received when any other `player` has joined the game
        client should add the player to the list of known players
        '''
        self._players.append(player)
        self.send_client('other_player_join', player)

    def dispatch_top_server_message(self):
        try: #socket version is non-blocking so we need to check for incomplete sends
            if self._conn.unfinished_send():
                if not self._conn.continue_send():
                    return
        except AttributeError: #multiprocessing version is blocking so those methods don't exist
            pass
        print('sending to server',self._server_message_queue[0][0])
        self._conn.send(self._server_message_queue.pop(0))

    def dispatch_top_client_message(self):
        print('sending to client',self._client_message_queue[0][0])
        self._pipe.send_bytes(cPickle.dumps(self._client_message_queue.pop(0), -1))

    def send_client(self, message, *args):
        self._client_message_queue.append((message, args))

class ClientServerConnection(object):
    '''
    represent a client-side connection to the minepy Multiplayer Server
    manages connections from players and handles data requests

    Maintains the following databases
        block information (delta from what the terrain generator produces)
        player information (unique id/name, location, velocity)
    '''
    def __init__(self, controller_pipe, SERVER_IP):
        self.handler = ClientServerConnectionHandler(controller_pipe, SERVER_IP)
        self.world = world_loader.World()
        #TODO: could use a decorator, though I think this is more readable
        self.handler.register_function('player_set_name',self.player_set_name)
        self.handler.register_function('player_set_postion',self.player_set_position)
        self.handler.register_function('player_set_block',self.player_set_block)
        self.handler.register_function('sector_blocks_changed',self.sector_blocks_changed)
        self.handler.communicate_loop()

    def player_set_name(self, player, name):
        '''
        `player` has changed their name to `name`
        client should update the name of that player
        '''
        player.name = name
        self.handler.send_client('player_set_name', player)

    def player_set_position(self, player, position):
        '''
        `player` has moved to `position`
        client should move (and display) player at the new position
        '''
        player.position = position
        self.handler.send_client('player_set_position', player)

    def player_set_block(self, player, position, block):
        '''
        `player` has added `block` at `position`
        client should add that block to its map
        '''
        for spos, vt_data, blocks in self.world.set_block(position, block):
            self.handler.send_client('sector_blocks', spos, blocks, vt_data)
        
    def sector_blocks_changed(self, player, sector_position, sector_blocks_delta):
        '''
        server has sent the player changed blocks in `sector_position`
        client should repalce its delta to the mapgen terrain with this one
        '''
        #vt_data, blocks = self.world.update_block_data(sector_position, sector_blocks_delta)
        vt_data, blocks = self.world.get_sector_data(sector_position, sector_blocks_delta)
        self.handler.send_client('sector_blocks', sector_position, blocks, vt_data)

def _start_server_connection(controller_pipe, SERVER_IP):
    conn = ClientServerConnection(controller_pipe, SERVER_IP)

class ClientServerConnectionProxy(object):
    def __init__(self, SERVER_IP = 'localhost'):
        self.pipe, _pipe = multiprocessing.Pipe()
        proc = multiprocessing.Process(target = _start_server_connection, args = (_pipe, SERVER_IP))
        proc.start()

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


def start_server_connection(SERVER_IP):
    return ClientServerConnectionProxy(SERVER_IP)

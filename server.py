# standard library imports
import time
import numpy
import select
try:
    import cPickle as pickle
except ImportError:
    import pickle
import msocket
import socket
import sys
import traceback

from players import Player, ClientPlayer
from config import SERVER_IP, SERVER_PORT
import world_db
    
def get_network_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.connect(('<broadcast>', 0))
    return s.getsockname()[0]
        
class ServerConnectionHandler(object):
    '''
    Handles the low level connection handling details of the multiplayer server
    '''
    def __init__(self):
        print('starting server at %s:%i'%(SERVER_IP,SERVER_PORT))
        self.listener = msocket.Listener(SERVER_IP, SERVER_PORT)
        self.players = []
        self.fn_dict = {}

    def register_function(self, name, fn):
        self.fn_dict[name]=fn

    def call_function(self,name,*args):
        return self.fn_dict[name](*args)

    def connections(self):
        return [p.conn for p in self.players]

    def connections_with_comms(self):
        try: #socket version is non-blocking so we need to check for incomplete sends
            return [p.conn for p in self.players if len(p.comms_queue)>0 or p.conn.unfinished_send()]
        except AttributeError: #multiprocessing version is blocking
            return [p.conn for p in self.players if len(p.comms_queue)>0]

    def player_from_connection(self, conn):
        for p in self.players:
            if conn == p.conn:
                return p

    def accept_connection(self):
        conn = self.listener.accept()
        player = Player(conn)
        self.players.append(player)
        return player

    def serve(self):
        alive = True
        while alive:
            r,w,x = select.select([self.listener] + self.connections(), self.connections_with_comms(), [])
            accept_new = True
            for p in self.players:
                if p.conn in r:
                    print('r select for ',p.id,p.name)
                    accept_new = False
                    try:
                        result = p.conn.recv()
                        if result is not None:
                            msg, data = result
                            print('received %s from player %i (%s)'%(msg, p.id, p.name))
                            print(data)
                            if msg == 'quit':
                                alive = False
                            else:
                                try:
                                    self.call_function(msg, p, *data)
                                except Exception as ex:
                                    traceback.print_exc()
                    except EOFError:
                        #TODO: Could allow a few retries before dropping the player
                        print('Disconnect due to EOF error on connection for player %i (%s)'%(p.id,p.name))
                        p.conn.close()
                        self.players.remove(p)
            for p in self.players:
                if p.conn in w:
                    print('w select for ',p.id,p.name)
                    self.dispatch_top_message(p)
            if accept_new and self.listener in r:
                p = self.accept_connection()
                print('connected new player with id %i'%(p.id,))
                self.queue_for_player(p, 'connected', ClientPlayer(p), [ClientPlayer(ap) for ap in self.players])
                self.queue_for_others(p, 'other_player_join', ClientPlayer(p))
        self.listener.close()

    ##TODO: queue calls should collapse similar calls (e.g. multiple block adds in the same sector)
    ##TODO: prioritize calls
    def queue_for_player(self, player, message, *data):
        player.comms_queue.append([message, player.id, data])

    def queue_for_others(self, player, message, *data):
        for p in self.players:
            if p != player:
                p.comms_queue.append([message, player.id, data])

    def queue_for_all_players(self, player, message, *data):
        for p in self.players:
            p.comms_queue.append([message, player.id, data])

    def other_players(self, player):
        return [p for p in self.players if p != player]

    def dispatch_top_message(self, player):
        try: #socket version is non-blocking so we need to check for incomplete sends
            if player.conn.unfinished_send():
                if not player.conn.continue_send():
                    return
        except AttributeError: #multiprocessing version is blocking
            pass
        print('sending %s to %i (%s)'%(player.comms_queue[0][0], player.id, player.name))
        #player.conn.send_bytes(pickle.dumps(player.comms_queue.pop(0), -1))
        player.conn.send(player.comms_queue.pop(0))

class Server(object):
    '''
    minepy Multiplayer Server
    manages connections from players and handles data requests

    Maintains the following databases
        block information (delta from what the terrain generator produces)
        player information (unique id/name, location, velocity)

    Server Messages (client must have handlers for these)
        connected(players)
            notifies the player that just connect with a list of all `players`
        other_player_join(player)
            notifies all other players that `player` has joined
        player_set_name(player, name)
            notifies all players that `player` is using name `name`
        player_set_position(player, position)
            notifies all other players that `player` is at `position`
        player_set_block(player, position, block)
            notifies all players that `player` has set `block` at `position`
        updated_sector_blocks(sector_pos, sector_blocks_delta)
            sends `sector_blocks_delta` to the player that requested it
    '''
    def __init__(self):
        self.handler = ServerConnectionHandler()
        self.world_db = world_db.World()
        #TODO: could use a decorator to avoid explicit registration, though I think this is more readable
        self.handler.register_function('set_name',self.set_name)
        self.handler.register_function('set_postion',self.set_position)
        self.handler.register_function('set_block',self.set_block)
        self.handler.register_function('sector_blocks',self.sector_blocks)
        self.handler.register_function('l_get_sector_blocks',self.l_get_sector_blocks)
        self.handler.register_function('l_get_seed',self.l_get_seed)
        try:
            self.handler.serve()
        except KeyboardInterrupt:
            print('Received keyboard interrupt')
            print('Shutting down...')
            #TODO: Notify players that we're shutting down

    def set_name(self, player, name):
        '''
        sets the unique name for the player
        player's client must wait for the confirmation player_set_name message from
        the server before assuming the name
        '''
        used = False
        for op in self.handler.other_players(player):
            if op.name == name:
                used = True
        if not used:
            player.name = data
        self.handler.queue_for_all_players(player, 'player_set_name', player.name)

    def set_position(self, player, position):
        '''
        sets the `position` tuple for the `player`
        confirmation is not required, player_set_position message is
        broadcast only to other players
        '''
        player.position = position
        self.handler.queue_for_others(player, 'player_set_position', position)

    def set_block(self, player, position, block):
        '''
        sets the block at `position` to the id given by `block`
        player's client must wait for the confirmation `player_set_name` message from
        the server before assuming the block has been set (to avoid synchronization issues
        when multiple players set the same block)
        '''
        self.world_db.set_block(position, block)
        self.handler.queue_for_others(player, 'player_set_block', position, block)

    def sector_blocks(self, player, sector_pos):
        '''
        request by `player` for the changed blocks in `sector_pos`
        data will be sent with the message `sector_blocks`
        '''
        blocks = self.world_db.get_sector_data(sector_pos)
        self.handler.queue_for_player(player, 'sector_blocks_changed', sector_pos, blocks)

    def l_get_sector_blocks(self, player, sector_pos):
        '''
        request by `player`'s loader for the changed blocks in `sector_pos`
        data will be sent with the message `sector_blocks`
        '''
        blocks = self.world_db.get_sector_data(sector_pos)
        self.handler.queue_for_player(player, 'l_sector_blocks_changed', sector_pos, blocks)

    def l_get_seed(self, player):
        self.handler.queue_for_player(player, 'l_seed', self.world_db.get_seed())


if __name__ == '__main__':
    #TODO: use a config file for server settings
    #TODO: use argparse module to override default server settings
    SERVER_IP = 'localhost'
    if len(sys.argv)>1:
        if sys.argv[1] == 'LAN':
            SERVER_IP = get_network_ip()
    s = Server()

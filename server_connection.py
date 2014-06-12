try:
    import cPickle as pickle
except ImportError:
    import pickle
import multiprocessing
import msocket
import select
import traceback

from config import SERVER_PORT
from players import Player
import world_loader

import logging
logging.basicConfig(level = logging.INFO)
def sconn_log(*args):
    logging.log(logging.INFO,*args)

class ClientServerConnectionHandler(object):
    '''
    Handles the connection to the multiplayer server and routes messages between client/loader and server
    '''
    def __init__(self, client_pipe, loader_pipe, SERVER_IP):
        sconn_log('connecting to server at %s:%i',SERVER_IP,SERVER_PORT)
#        self._conn = multiprocessing.connection.Client(address = (SERVER_IP,SERVER_PORT), authkey = 'password')
        self._conn = msocket.Client(SERVER_IP,SERVER_PORT)
        self._pipe = client_pipe
        self._loader_pipe = loader_pipe
        self._server_message_queue = []
        self._client_message_queue = []
        self._loader_message_queue = []
        self._players = []

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
            if len(self._loader_message_queue)>0:
                w.append(self._loader_pipe)
            r,w,x = select.select([self._conn, self._pipe, self._loader_pipe], w, [])
            if self._conn in r:
                try:
                    sconn_log('msg from server')
                    result = self._conn.recv()
                    if result is not None:
                        msg, pid, data = result
                        sconn_log('msg from server %s',msg)
                        if msg == 'connected':
                            self.connected(pid, *data)
                        elif msg == 'other_player_join':
                            self.other_player_join(pid, *data)
                        else:
                            p = self.player_from_id(pid)
                            if msg.startswith('l_'):
                                self._loader_message_queue.append((msg, data))
                            else:
                                self._client_message_queue.append((msg, data))
                except EOFError:
                    ##TODO: disconnect from server / tell parent / try to reconnect
                    alive = False
            if self._pipe in r:
                msg, data = self._pipe.recv()
                sconn_log('msg from client %s',msg)
                if msg == 'quit':
                    ##TODO: disconnect from server
                    alive = False
                self._server_message_queue.append((msg, data))
            if self._loader_pipe in r:
                msg, data = self._loader_pipe.recv()
                sconn_log('msg from loader %s',msg)
                self._server_message_queue.append((msg, data))
            if self._conn in w:
                sconn_log('msg from loader %s',msg)
                self.dispatch_top_server_message()
            if self._pipe in w:
                sconn_log('msg from loader %s',msg)
                self.dispatch_top_client_message()
            if self._loader_pipe in w:
                sconn_log('msg from loader %s',msg)
                self.dispatch_top_loader_message()
        self._conn.close()
        self._pipe.close()
        self._loader_pipe.close()

    def connected(self, player_id, player, players):
        '''
        received when the `player` has successfully joined the game
        '''
        self._players = players
        sconn_log('connected %i(%s)', player_id, str(players))
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
        sconn_log('sending to server %s',self._server_message_queue[0][0])
        self._conn.send(self._server_message_queue.pop(0))

    def dispatch_top_client_message(self):
        sconn_log('sending to client %s',self._client_message_queue[0][0])
        self._pipe.send_bytes(pickle.dumps(self._client_message_queue.pop(0), -1))

    def dispatch_top_loader_message(self):
        sconn_log('sending to loader %s',self._loader_message_queue[0][0])
        self._loader_pipe.send_bytes(pickle.dumps(self._loader_message_queue.pop(0), -1))

    def send_client(self, message, *args):
        self._client_message_queue.append((message, args))


def _start_server_connection(client_pipe, loader_pipe, SERVER_IP):
    conn = ClientServerConnectionHandler(client_pipe, loader_pipe, SERVER_IP)
    conn.communicate_loop()

class ClientServerConnectionProxy(object):
    def __init__(self, SERVER_IP = 'localhost'):
        self.pipe, _pipe = multiprocessing.Pipe()
        self.loader_pipe, _loader_pipe = multiprocessing.Pipe()
        self.proc = multiprocessing.Process(target = _start_server_connection, args = (_pipe, _loader_pipe, SERVER_IP))
        self.proc.start()

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

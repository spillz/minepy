
def get_network_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.connect(('<broadcast>', 0))
    return s.getsockname()[0]

def broadcast_server(port):
    #UDP server responds to broadcast packets
    #you can have more than one instance of these running
    import socket
    address = ('', port)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    server_socket.bind(address)

    while True:
        print "Listening"
        recv_data, addr = server_socket.recvfrom(2048)
        print addr,':',recv_data
        server_socket.sendto("*"+recv_data, addr)

def broadcast_client(port):
    #UDP client broadcasts to server(s)
    import socket

    address = ('<broadcast>', port)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    data = "Request"
    client_socket.sendto(data, address)
    while True:
        recv_data, addr = client_socket.recvfrom(2048)
        print addr,recv_data

use_multiprocessing = True

if use_multiprocessing:
    
    import multiprocessing.connection

    class Listener(multiprocessing.connection.Listener):
        def __init__(self, ip, port):
            multiprocessing.connection.Listener.__init__(self, address = (ip, port), authkey = 'password')
            
        def fileno(self):
            return self._listener._socket.fileno()

    def Client(ip, port):
        return multiprocessing.connection.Client(address = (ip, port), authkey = 'password')

else:
    
    import socket
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    import struct

    fmt = 'l'
    lencoder = lambda value: struct.pack(fmt,value)
    lendecoder = lambda string: struct.unpack(fmt,string)[0]
    lenc = struct.calcsize(fmt)


    class Connection(object):
        def __init__(self, sock, addr):
            self._sock = sock
            self._sock.setblocking(0)
            self._addr = addr
            self.send_buffer= ''
            self.recv_buffer = ''
            self.recv_count_buffer = ''
            self.recv_count = 0
            self.recv_finished = True

        def fileno(self):
            return self._sock.fileno()

        def unfinished_send(self):
            return len(self.send_buffer) > 0

        def continue_send(self):
            if len(self.send_buffer) == 0:
                return True
            wrote = self._sock.send(self.send_buffer)
            self.send_buffer = self.send_buffer[wrote:]
            return len(self.send_buffer) == 0

        def continue_recv(self):
            if self.recv_finished:
                return
            self.recv_finished = False
            if len(self.recv_count_buffer) < lenc:
                self.recv_count_buffer += self._sock.recv(lenc)
                if len(self.recv_count_buffer) == lenc:
                    self.recv_count = lendecoder(self.recv_count_buffer)
                elif len(self.recv_count_buffer) == 0:
                    raise EOFError
                else:
                    return
            ##TODO: check length of count raise exception if too big
            while len(self.recv_buffer) < self.recv_count: ##TRY TO LOOP UNTIL ALL BYTES ARE RECEIVED
                remaining = self.recv_count - len(self.recv_buffer)
                prev_read = len(self.recv_buffer)
                self.recv_buffer += self._sock.recv(min(remaining,4096))
                if len(self.recv_buffer) -prev_read < min(remaining, 4096):
                    return
            data = pickle.loads(self.recv_buffer)
            self.recv_finished = True
            self.recv_buffer = ''
            self.recv_count_buffer = ''
            return data

        def send(self, data):
            datastream = pickle.dumps(data, -1)
            count = len(datastream)
            ##TODO: check length of count raise exception if too big
            self.send_buffer = lencoder(count) + datastream
            return self.continue_send()

        def recv(self):
            if not self.recv_finished:
                return self.continue_recv()
            self.recv_count_buffer = ''
            self.recv_buffer =''
            self.recv_finished = False
            return self.continue_recv()

        def close(self):
            self._sock.close()

    class Listener(object):
        def __init__(self, ip, port):
            self._sock = socket.socket()
            self._sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) 
            self._sock.setblocking(0)
            self._sock.bind((ip, port))
            self._sock.listen(5)
            
        def fileno(self):
            return self._sock.fileno()

        def accept(self):
            return Connection(*self._sock.accept())

    class Client(Connection):
        def __init__(self, ip, port):
            sock = socket.socket()
            sock.connect((ip, port))
            Connection.__init__(self,sock,(ip, port))

        

counter = 0

class Player(object):
    def __init__(self, conn):
        global counter
        self.conn = conn
        self.id = counter
        counter+=1
        self.name = 'FRED'
        self.position = (0,0,0)
        self.comms_queue = []

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

class ClientPlayer(object):
    def __init__(self, player):
        self.id = player.id
        self.name = player.name
        self.position = player.position
    
    def __repr__(self):
        return self.name
        
    def __str__(self):
        return self.name

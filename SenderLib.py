import socket
import os, re

class Sender:

    def __init__(self, host="127.0.0.1", port=25002):
        self.host, self.port = host, port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, self.port))

    startPos = [1, 0, 0]
    turn = False

    def send_data_both(self, lift):
        self.startPos[1] = 1 if lift else 0
        self.startPos[2] = 1 if lift else 0
        pos_string = ','.join(map(str, self.startPos))
        # print(pos_string)
        self.sock.sendall(pos_string.encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        print(receivedData)

    def send_data(self, lift_r, lift_l):
        self.startPos[1] = 1 if lift_l else 0
        self.startPos[2] = 1 if lift_r else 0
        pos_string = ','.join(map(str, self.startPos))
        self.sock.sendall(pos_string.encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        # print(receivedData)


def get_local_addresses():
    full_results = [re.findall('^[\w\?\.]+|(?<=\s)\([\d\.]+\)|(?<=at\s)[\w\:]+', i) for i in os.popen('arp -a')]
    final_results = [dict(zip(['IP', 'LAN_IP', 'MAC_ADDRESS'], i)) for i in full_results]
    print(str(full_results))
    final_results = [{**i, **{'LAN_IP': i['LAN_IP'][1:-1]}} for i in final_results]

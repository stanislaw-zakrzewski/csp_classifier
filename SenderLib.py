import os
import re
import socket
import json
from collections import namedtuple


class GameState:
    buttonPressed = False


class GameControl:
    movement = True
    left = False
    right = False
    applyMode = True
    mode = 4

    dataAcquisition = True

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class Sender:

    def __init__(self, host="127.0.0.1", port=25002):
        self.host, self.port = host, port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, self.port))

    turn = False

    def send_to_tcp(self, control) -> GameState:
        self.sock.sendall(control.to_json().encode("UTF-8"))
        received_data = self.sock.recv(1024).decode("UTF-8")
        # print(str(received_data))
        return json2obj(received_data)

    def send_data(self, control) -> GameState:
        control.movement = True
        return self.send_to_tcp(control)

    def get_state(self, control) -> GameState:
        control.movement = False
        return self.send_to_tcp(control)


def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())


def json2obj(data) -> GameState: return json.loads(data, object_hook=_json_object_hook)


def get_local_addresses():
    full_results = [re.findall('^[\w\?\.]+|(?<=\s)\([\d\.]+\)|(?<=at\s)[\w\:]+', i) for i in os.popen('arp -a')]
    final_results = [dict(zip(['IP', 'LAN_IP', 'MAC_ADDRESS'], i)) for i in full_results]
    print(str(full_results))
    final_results = [{**i, **{'LAN_IP': i['LAN_IP'][1:-1]}} for i in final_results]

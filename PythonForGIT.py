import socket
import time

host, port = "127.0.0.1", 25002
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

startPos = [1, 0, 0]
turn = False
while True:
    time.sleep(5)
    posString = ','.join(map(str, startPos))
    print(posString)

    sock.sendall(posString.encode("UTF-8"))
    receivedData = sock.recv(1024).decode("UTF-8")
    print(receivedData)
    # startPos[1] = 1 if turn else 0
    startPos[2] = 0 if turn else 1
    turn = not turn

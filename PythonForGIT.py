from time import sleep
import SenderLib

#EEGLAB:
# EEGlaptop: 192.168.50.100
# Gogle VR 1: 192.168.50.101
# Gogle VR 2: 192.168.50.102

host, port = "127.0.0.1", 25002
# host, port = "192.168.132.6", 25002

sender = SenderLib.Sender(host, port)
turn = False

gameControl = SenderLib.GameControl()
gameControl.applyMode = True
gameControl.mode = 4
gameControl.dataAcquisition = True

while True:
    print("Movement" if turn else "Rest")
    gameControl.left = turn
    x = sender.send_data(gameControl)
    #print(str())
    turn = not turn
    sleep(6)

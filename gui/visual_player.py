import os
import time
import tkinter as tk
from threading import *

import vlc


class Screen(tk.Frame):

    '''
    Screen widget: Embedded video player from local or youtube
    '''

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, bg='black')
        self.parent = parent
        # Creating VLC player
        self.instance = vlc.Instance('--input-repeat=999999')
        self.player = self.instance.media_player_new()

        # def change(rt):
        #     n = 0
        #     while True:
        #         print('leci', n)
        #         n+=1
        #         time.sleep(1/100)
        #         # Media = rt.instance.media_new("commands//visual_commands//rest.png")
        #         # rt.player.set_media(Media)
        #         # rt.player.play()
        #
        # audio_thread = Thread(target=change, args=(self,))  # create thread
        # audio_thread.start()

    def GetHandle(self):
        # Getting frame ID
        return self.winfo_id()

    def play(self, _source):
        # Function to start player from given source
        Media = self.instance.media_new(_source)
        Media.get_mrl()
        self.player.set_media(Media)

        self.player.set_hwnd(self.winfo_id())
        self.player.play()

    def stop(self):
        Media = self.instance.media_new("commands//visual_commands//rest.png")
        self.player.set_media(Media)
        self.player.play()



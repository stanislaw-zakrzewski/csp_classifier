from threading import Thread

#from playsound import playsound

from commands.commands import Commands

import wave
import sys

import pyaudio

CHUNK = 1024

class AudioCommands(Commands):
    def __init__(self):
        self.previous_command = False
        self.blocked = False
        self.left_audioFrames = self.loadAudio('commands//sound_commands//lewo.wav')
        self.right_audioFrames = self.loadAudio('commands//sound_commands//prawo.wav')
        self.rest_audioFrames = self.loadAudio('commands//sound_commands//brak.wav')
        self.movement_audioFrames = self.loadAudio('commands//sound_commands//ruch.wav')
        self.pause_audioFrames = self.loadAudio('commands//sound_commands//pauza.wav')
        self.end_audioFrames = self.loadAudio('commands//sound_commands//koniec.wav')

    def loadAudio(self, fname):
        audioFrames = list()
        with wave.open(fname, 'rb') as wf:
            while len(data := wf.readframes(CHUNK)):
                audioFrames.append(data)
        self.theSampWidth = wf.getsampwidth()
        self.theNumChannels = wf.getnchannels()
        self.theFrameRate = wf.getframerate()
        return audioFrames

    def play_audio_command(self, command_code):
        if command_code == 'left':
            self.left()
        elif command_code == 'right':
            self.right()
        elif command_code == 'rest':
            self.rest()
        elif command_code == 'movement':
            self.movement()
        elif command_code == 'pause':
            self.pause()
        elif command_code == 'end':
            self.end()
        self.blocked = False
        self.previous_command = command_code

    def perform_command(self, command_code):
        if not self.blocked and command_code != self.previous_command:
            self.blocked = True
            self.previous_command = command_code
            audio_thread = Thread(target=self.play_audio_command, args=(command_code,))  # create thread
            audio_thread.start()

    def doPlay(self, audioFrames):
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(self.theSampWidth), channels=self.theNumChannels, rate=self.theFrameRate, output=True)
        for frame in audioFrames:
            stream.write(frame)
        stream.close()
        p.terminate()

    def left(self):
        self.doPlay(self.left_audioFrames)

    def right(self):
        self.doPlay(self.right_audioFrames)

    def rest(self):
        self.doPlay(self.rest_audioFrames)

    def movement(self):
        self.doPlay(self.movement_audioFrames)

    def pause(self):
        self.doPlay(self.pause_audioFrames)

    def end(self):
        self.doPlay(self.end_audioFrames)

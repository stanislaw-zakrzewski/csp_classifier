from threading import Thread

from playsound import playsound

from commands.commands import Commands


class AudioCommands(Commands):
    def __init__(self):
        self.previous_command = False
        self.blocked = False

    def play_audio_command(self, command_code):
        if command_code == 'left':
            self.left()
        elif command_code == 'right':
            self.right()
        elif command_code == 'rest':
            self.rest()
        elif command_code == 'movement':
            self.movement()
        self.blocked = False
        self.previous_command = command_code

    def perform_command(self, command_code):
        if not self.blocked and command_code != self.previous_command:
            self.blocked = True
            self.previous_command = command_code
            audio_thread = Thread(target=self.play_audio_command, args=(command_code,))  # create thread
            audio_thread.start()

    def left(self):
        playsound('commands//sound_commands//lewo.mp3')

    def right(self):
        playsound('commands//sound_commands//prawo.mp3')

    def rest(self):
        playsound('commands//sound_commands//brak.mp3')

    def movement(self):
        playsound('commands//sound_commands//ruch.mp3')

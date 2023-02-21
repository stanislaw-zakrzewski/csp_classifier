from commands.commands import Commands


class TextCommands(Commands):
    def __init__(self):
        self.previous_command = False

    def perform_command(self, command_code):
        if self.previous_command != command_code:
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
            self.previous_command = command_code

    def left(self):
        print('left')

    def right(self):
        print('right')

    def rest(self):
        print('rest')

    def movement(self):
        print('movement')

    def pause(self):
        print('pause')

    def end(self):
        print('end')

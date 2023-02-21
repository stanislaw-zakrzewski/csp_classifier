from commands.audio_commands import AudioCommands
from commands.text_commands import TextCommands
from commands.visual_commands import VisualCommands
from config import command_type


def get_commands():
    commands = TextCommands()
    if command_type == 'audio':
        commands = AudioCommands()
    if command_type == 'visual':
        commands = VisualCommands()
    return commands

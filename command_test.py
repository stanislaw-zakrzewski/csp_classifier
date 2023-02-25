import time
from commands.audio_commands_pyaudio import AudioCommands
from commands.visual_commands import VisualCommands

vc = AudioCommands()
vc.perform_command('pause')
time.sleep(2)
vc.perform_command('movement')
time.sleep(8)
vc.perform_command('pause')
time.sleep(2)
vc.perform_command('left')
time.sleep(8)
vc.perform_command('pause')
time.sleep(2)
vc.perform_command('right')
time.sleep(8)
vc.perform_command('pause')
time.sleep(2)
vc.perform_command('rest')
time.sleep(8)
vc.perform_command('end')

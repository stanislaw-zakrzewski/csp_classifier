import os
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import time
class Screen(tk.Frame):
   '''
   Screen widget: Embedded OpenCV video player for seamless looping and switching.
   No external system libraries (like VLC/MPV DLLs) required.
   '''
   def __init__(self, parent, *args, **kwargs):
       tk.Frame.__init__(self, parent, bg='black', *args, **kwargs)
       self.parent = parent
       # We use a Label to display the video frames
       self.video_label = tk.Label(self, bg='black')
       self.video_label.pack(fill=tk.BOTH, expand=True)
       self.cap = None
       self.is_video = False
       self.delay = 33  # Default delay (~30 FPS)
       self.media_map = {
           'rest': 'commands/visual_commands/rest.png',
           'break': 'commands/visual_commands/pause.jpg',
           'movement': 'commands/visual_commands/movement.mov'
       }
       # Start the Tkinter frame update loop immediately
       self.update_frame()
   def play(self, _source):
       filepath = self.media_map.get(_source, _source)
       ext = os.path.splitext(filepath)[1].lower()
       if ext in ['.png', '.jpg', '.jpeg']:
           # --- Handle Static Images ---
           self.is_video = False
           # Clean up the old video capture if it was running
           if self.cap is not None:
               self.cap.release()
               self.cap = None
           # Read and display the image once
           frame = cv2.imread(filepath)
           if frame is not None:
               self.display_frame(frame)
       else:
           # --- Handle Videos seamlessly ---
           new_cap = cv2.VideoCapture(filepath)
           if new_cap.isOpened():
               self.is_video = True
               # Seamless transition: Swap to the new video before destroying the old one
               if self.cap is not None:
                   self.cap.release()
               self.cap = new_cap
               # Automatically calculate playback speed based on video metadata
               fps = self.cap.get(cv2.CAP_PROP_FPS)
               if fps > 0:
                   # Rename this to target_delay to act as our baseline
                   self.target_delay = int(1000 / fps)
               else:
                   self.target_delay = 33
   def stop(self):
       # Route the stop command to display the 'rest' image
       self.play('rest')

   def display_frame(self, frame):
       """Converts an OpenCV frame, stretches it, and updates the label."""
       win_width = self.winfo_width()
       win_height = self.winfo_height()
       if win_width > 1 and win_height > 1:
           # Changed to INTER_LINEAR - it is much faster for real-time stretching
           frame = cv2.resize(frame, (win_width, win_height), interpolation=cv2.INTER_LINEAR)
       cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       img = Image.fromarray(cv2image)
       imgtk = ImageTk.PhotoImage(image=img)
       self.video_label.imgtk = imgtk
       self.video_label.configure(image=imgtk)

   def update_frame(self):
       """Native Tkinter loop with dynamic delay to maintain true FPS."""
       # Start a stopwatch
       start_time = time.perf_counter()
       if self.is_video and self.cap is not None:
           ret, frame = self.cap.read()
           if not ret:
               self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
               ret, frame = self.cap.read()
           if ret:
               self.display_frame(frame)
       # Stop the watch and calculate how many milliseconds the processing took
       processing_time = int((time.perf_counter() - start_time) * 1000)
       # Subtract the processing time from our target delay.
       # Use max(1, ...) to ensure we never pass a negative number to Tkinter.
       # If processing took longer than the target delay, it will move to the next frame in 1ms.
       actual_delay = max(1, getattr(self, 'target_delay', 33) - processing_time)
       self.after(actual_delay, self.update_frame)
   def terminate(self):
       """Cleanup function to release the camera/file lock on exit."""
       if self.cap is not None:
           self.cap.release()
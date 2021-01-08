from moviepy.editor import *

clip = VideoFileClip("Full_comp.mp4").resize(0.7)
   
clip.write_gif("use_your_head.gif", fps=15)
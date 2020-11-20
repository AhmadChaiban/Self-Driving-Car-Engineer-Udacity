import cv2
from cv2 import dnn_superres
from glob import glob as globlin ## 7bb <3
import progressbar

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the desired model
path = "EDSR_x3.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 2)

paths = globlin('./video/*.*')
with progressbar.ProgressBar(max_value=len(paths)) as bar:
    for index, path in enumerate(paths):
        # Read image
        image = cv2.imread(path)
        # Upscale the image
        result = sr.upsample(image)
        # Save the image
        cv2.imwrite(path.replace('video', 'video2'), result)
        bar.update(index)


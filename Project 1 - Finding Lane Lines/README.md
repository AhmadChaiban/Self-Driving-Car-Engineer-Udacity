# Project 1: Finding Lane Lines

### Overview: 

In this Udacity project, I was tasked to find lanes, using Computer Vision techniques, from several videos that were provided. 

### Files: 

- **P1.ipynb** is the notebook that was initially given by the project. It was filled out at the end (it's better for presentation).
- **lane_detect.py** is the file where the image pipeline resides. It is the "main".
- **helpers.py** functions that include all the image processing techniques.

To run the project, simply run **lane_detect.py**, the videos will show up in test_videos_output.

### Pipeline: 

1. Video is processed per frame. 
2. A region of interested is created in order to drastically minimize error. 
3. Each frame is converted to grayscale.  
4. Gaussian Blurring with Kernel Size 1 is then applied to each frame. 
5. Canny edge detection is then applied with a low threshold = 100 and high threshold = 200
6. Since the region of interest shows up in the canny edge detection, it is cropped out of the image. 
7. Hough lines are drawn up with the following parameters:
    - rho=1,
    - theta=1*np.pi/180,
    - threshold=15,
    - min_line_len=40,
    - max_line_gap=25
    - During this process, the polyfit function was used to fit a polynomial to the points (x,y) coordinates 
    of the points. 
8. The image is displayed with the lane estimation lines. 

### Results:

Initial images. The algorithm seems to be performing accurately: 

<center>
    <img src="result_gifs/solidWhiteRight.gif"/>
    <p> </p>
    <img src="result_gifs/solidYellowLeft.gif"/>
</center>

### Challenge:

There is a lot of noise in this image, not quite there but almost. This is one of the shortcomings of the algorithm. The lines don't 
adjust to turns, and some more noise reduction is required. 

<center>
    <img src="result_gifs/challenge.gif"/>
</center>

### Extra image:

This is a sample of a dash cam video on youtube, it performed relatively well, however, when the person is changing lanes, the 
algorithm does not account for it. The lane detection resumes correctly when the vehicle has completely changed lanes:

<center>
    <img src="result_gifs/Toronto-youtube.gif"/>
</center>
<br>
Really fun project. Took my time with it!
<h1 style="color: #3a87ad">Udacity Project 4 - Behavioral Cloning</h1>

<h2 style="color: #3a87ad">Introduction</h2>

In this project, I was tasked with teaching a Neural Network my own driving pattern in Udacity's simulator. The simulator 
collected images of my driving that I used to train and test several neural networks. I tried LeNet, InceptionV3 with 
some modifications and a model architecture outlined by nVidia in this 
<a href="https://developer.nvidia.com/blog/deep-learning-self-driving-cars/?ncid=afm-chs-44270&ranMID=44270&ranEAID=a1LgFw09t88&ranSiteID=a1LgFw09t88-eRZ5swEwc1zdTIYasZfm8A" target="_blank">
paper</a>. The nVidia network scored the highest, with an **r2 score = 0.64**, and result can be seen in the **track1_video.mp4** file. 

<h2 style="color: #3a87ad">Simulator, cameras and images</h2>

The simulator allowed the user to drive the car using specific steering controls with the mouse in order to capture the 
steering angle more accurately. Here is a sample of the simulator:

<center><img src="./image_docs/simulator_sample.png"/></center> 

<br>

With the following images representing the different camera angles, left, center and right respectively:

<center>
<img src="./image_docs/left_2020_11_19_23_15_25_899.jpg" width="290">
<img src="./image_docs/center_2020_11_19_23_15_25_899.jpg" width="290"/>
<img src="./image_docs/right_2020_11_19_23_15_25_899.jpg" width="290"/>
</center>
<br>
Which were combined in order to train the model to readjust its driving if it starts to approach failure or go off track.

<h2 style="color: #3a87ad">Preprocessing</h2>

First, the images and their respective training angles were 

<h2 style="color: #3a87ad">nVidia Network</h2>

<h2 style="color: #3a87ad">Results</h2>

<h1 style="color:#3a7aad">Localization Project Introduction</h1>

In this project you will implement a 2 dimensional particle filter in C++. Your particle filter will be given a map and some initial localization information (analogous to what a GPS would provide). At each time step your filter will also get observation and control data.

You can find the project rubric here.

<h2 style="color:#3a7aad">Using GitHub and Creating Effective READMEs</h2>
If you are unfamiliar with GitHub, Udacity has a brief GitHub tutorial to get you started. Udacity also provides a more detailed free course on git and GitHub.

To learn about README files and Markdown, Udacity provides a free course on READMEs, as well.

GitHub also provides a tutorial about creating Markdown files.


<h2 style="color:#3a7aad">Particle Filter Project Visualizer</h2>

The Term 2 Simulator includes a graphical version of the Kidnapped Vehicle Project. Running the simulator you can see the path that the car drives along with all of its landmark measurements. (Note: If you choose to utilize the Workspace for this project, both the similator and related project repository are included therein, so you won't need to download/clone them locally except as desired).

Included in the Kidnapped Vehicle project Github repository are program files that allow you to set up and run c++ uWebSocketIO, which is used to communicate with the simulator. The simulator provides the script for the noisy position data, vehicle controls, and noisy observations. The script feeds back the best particle state.

The simulator can also display the best particle's sensed positions, along with the corresponding map ID associations. This can be extremely helpful to make sure transition and association calculations were done correctly. Below is a video of what it looks like when the simulator successfully is able to track the car to a particle. Notice that the green laser sensors from the car nearly overlap the blue laser sensors from the particle, this means that the particle transition calculations were done correctly.

<h3 style="color:#3a7aad">Download Links for Term 2 Simulator</h3>

https://github.com/udacity/self-driving-car-sim/releases

<h3 style="color:#3a7aad">Running the Program</h3>

1. Download the simulator and open it. In the main menu screen select Project 3: Kidnapped Vehicle.

2. Once the scene is loaded you can hit the START button to observe how the car drives and observes landmarks. At any time you can press the PAUSE button, to pause the scene or hit the RESTART button to reset the scene. Also the ARROW KEYS can be used to move the camera around, and the top left ZOOM IN/OUT buttons can be used to focus the camera. Pressing the ESCAPE KEY returns to the simulator main menu.

3. The Kidnapped Vehicle project Github repository README has more detailed instructions for installing and using c++ uWebScoketIO.

<h3 style="color:#3a7aad">Running the Kidnapped Vehicle project.</h3>

https://github.com/udacity/CarND-Kidnapped-Vehicle-Project
This workspace is designed to be a simple, easy to use environment in which you can code and run the Kidnapped Vehicle project. If you prefer to run the project on your computer, you can clone/fork the repository in your local setup.

For tips on workspace use, please review the Workspaces lesson from Term 1.

<h3 style="color:#3a7aad">Accessing and using the workspace:</h3>

* Navigate to the workspace node.
* Navigate to the repository CarND-Kidnapped-Vehicle-Project using menu on the left.
* Complete the TODO in particle_filter.cpp and particle_filter.h using the text editor in the workspace.
* Navigate to the project repository in the terminal.

The main program can be built and run by doing the following from the project top directory:

1. In the terminal execute ./clean.shto make sure you don't have old files in the directory.
2. In the terminal execute ./build.sh to build the project.
3. In the terminal execute ./run.sh to execute your solution.

All Project instructions can be found in the README.md (you can view the instruction in an easy-to-read format by visiting the previous link).
https://github.com/udacity/CarND-Kidnapped-Vehicle-Project/blob/master/README.md

Click on the "Simulator" button in the bottom of the Udacity workspace, which will open a new virtual desktop. You should see a "Simulator" icon on the virtual desktop. Double-click the "Simulator" icon in that desktop to start the simulator.

**Important:** You need to open a terminal before attempting to run the simulator.
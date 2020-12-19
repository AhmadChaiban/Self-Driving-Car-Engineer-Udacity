<h1 style="color: #3a7aad">Project 7: Path Planning and Highway Driving</h1>

The goal of this project was to build a path planner that creates smooth, safe trajectories for the car to follow. 
The highway track has other vehicles, all going different speeds and making lane changes, but approximately obeying 
the 50 MPH speed limit.

The car transmits its location, along with its sensor fusion data, which estimates the location of all the vehicles 
on the same side of the road. This data was considered when planning the vehicle's path. 

<h2 style="color: #3a7aad">Methodology and Cost Functions</h2>

The following steps were taken in order to create a smooth trajectory for the vehicle, allow it to change lanes, and 
prevent collisions with other vehicles:

1. 

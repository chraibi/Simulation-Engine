# Crowd-Simulation
Python learning project Oct 2024

![demo-gif](https://github.com/benw000/Crowd-Simulation/blob/main/demo.gif)

Files:
---

- pedestrian.py : contains Person class with force-based model methods
- basic_simulation.py : script that simulates the dynamics given the force based model from pedestrian.py
    - Run with terminal command: \
    ```python basic_simulation.py --num_people 10 --save_mp4 True```
- demo.mp4, demo.gif : demonstration video simulation
- learning_classes.py : script to learn classes and OOP
- VersionControl.md : markdown document with Git commands and explanation
- .gitignore : file telling Git to ignore output CSV and MP4 folders
- publish_all.sh : bash script to automatically push to main (Use carefully)

Aims:
----
- Create a crowd simulation script, which will take a number of pedestrians and model their movement with a force-based model
- This should take initial positions, a custom 2D environment, and update each pedestrians position and velocity at each timestep
- This should be done following Git and Python best practices, using class instances for each pedestrian. Use git terminal commands, and commit from develop branch to main.
- This should store information in an updated CSV file for each run
- At the end of the computation, the script should read the CSV by line and create an mp4 video

Extensions:
-----
- Create seperate child classes with different behaviours and different colours
- Extend function or class to have a torus border that flows from top to bottom, no walls
- Extend to using walls specified as a polygonal mesh of points and edges, work out distance from each
- Create evacuation study with doors for holes in the wall, people pass and are taken out of simulation
- Look at realistic timescales, movement speeds and thresholds etc
- Create seperate planetary gravitation simulation with same principals but RK45 instead of forward euler
- Create a pop up window where you click on a screen to initialise positions, and specify a velocity
- Port all of this to a HTML page which has a Python backend

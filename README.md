# Simulation Engine
## General simulation engine for force-based particle models, built from scratch in Python.
Oct 2024 \
Author: Ben Winstanley

----

### Predator-Prey Model


![birds-gif](https://github.com/benw000/Crowd-Simulation/blob/main/demo_videos/birds_demo.gif)

Run this :  ``` python general_simulation.py --type birds --num 50 --num2 5 --steps 200```

--- 

### N-body Gravitational Dynamics


![nbody-gif](https://github.com/benw000/Crowd-Simulation/blob/main/demo_videos/nbody_demo.gif)

Run this :  ``` python general_simulation.py --type nbody --num 30 --steps 100 ```

---

### Classroom Evacuation model


![evac-gif](https://github.com/benw000/Crowd-Simulation/blob/main/demo_videos/evac_demo.gif)

Run this :  ``` python general_simulation.py --type evac --num 40 --steps 200 ```

---




Files:
---

- classes.py : contains main Particle class, with Prey, Predator, Star, Human child classes, and Environment class with Wall child class.
- simulation_classes : module folder containing seperated files with each of these classes
- general_simulation.py : calls objects from classes.py, wraps up in simple time-stepping script in main(). Run with commands specified above.
- old_version : folder containing older version of project
- demo_videos : folder containing demo videos and gifs
- VersionControl.md : markdown document with Git commands and explanation
- .gitignore : file telling Git to ignore output CSV and MP4 folders
- publish_all.sh : bash script to automatically push to main (Use carefully)

Starting Aims:
----
- Create a crowd simulation script, which will take a number of pedestrians and model their movement with a force-based model
- This should take initial positions, a custom 2D environment, and update each pedestrians position and velocity at each timestep
- This should be done following Git and Python best practices, with object-oriented programming. Use git terminal commands, and commit from develop branch to main.
- This should store information in an updated CSV file for each run.
- At the end of the computation, the script should read the CSV by line and create an mp4 video.

Extensions:
-----
- Cloth simulation using perturbed SHM oscillator model for each particle in lattice.
- Evac sim fire instances
- Evac sim where we fuse force model with a neural network to work out velocity. Reinforcement learning with PyTorch to train:
    1. Selfish NN behaviour to minimise personal time until evacuated
    2. Collective NN behaviour to minimise total time until all evacuated
- If this works, do the same with birds in torus world, passing torus shortest dist and dirn. We then could have a GAN between predator and prey, both competing.
- Port all of this to a HTML page which has a Python backend
- Host HTML page 

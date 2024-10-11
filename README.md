# Crowd-Simulation
Python learning project Oct 2024

![demo-gif](https://github.com/benw000/Crowd-Simulation/blob/main/demo.gif)

Files:
---

- classes.py : contains main Particle class, with Prey, Predator, Star, Human child classes, and Environment class with Wall child class.
- general_simulation.py : calls objects from classes.py, wraps up in simple time-stepping script in main(). 
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
- Seperate into module structure with clear imports
- main() system input to determine model type and parameters.
- Cloth simulation using perturbed SHM oscillator model for each particle in lattice
- Evac sim fire instances
- Evac sim where we fuse force model with a neural network to work out velocity. Reinforcement learning with PyTorch to train:
    1. Selfish NN behaviour to minimise personal time until evacuated
    2. Collective NN behaviour to minimise total time until all evacuated
- If this works, do the same with birds in torus world, passing torus shortest dist and dirn. We then could have a GAN between predator and prey, both competing.
- Port all of this to a HTML page which has a Python backend
- Host HTML page 

# classes.py - generalising pedestrian.py

import numpy as np

counter = 0

class Particle:
    '''
    Parent class for particles in a 2D plane
    '''
    all = []

    walls_x_lim = 100
    walls_y_lim = 100

    # Class to describe each particle in simulation
    def __init__(self, 
                position: np.ndarray = None,
                velocity: np.ndarray = None) -> None:
        
        # Assign random position and stationary velocity if not given
        if position is None:
            self.position = np.array([np.random.rand(1)[0]*self.walls_x_lim,np.random.rand(1)[0]*self.walls_y_lim])
        if velocity is None:
            self.velocity = np.zeros(2)

        # Start with zero acceleration
        self.acceleration = np.zeros(2)

        # Add to set of all people
        global counter
        self.id = counter
        counter += 1
        Particle.all += [self]

    # Print statement for particle
    def __str__(self) -> str:
        return f"Particle {self.id}at position {self.position} with velocity {self.velocity}"

    # Debug statement for particle
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.id},{self.position},{self.velocity})"
    
    # Calculate squared euclidean distance between particles
    def dist(self,other) -> float:
        return np.sum((self.position-other.position)**2)
    
    # Calculate distance direction between particles
    def dirn(self,other) -> float:
        return (other.position-self.position)/np.sqrt(self.dist(other))


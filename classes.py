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
        '''
        Initiate generic particle, with optional position and velocity inputs.
        Start with random position, and zero velocity and zero acceleration.
        Set an ID value and then increment ID counter.
        '''
        # Assign random position and stationary velocity if not given
        if position is None:
            self.position = np.array([np.random.rand(1)[0]*self.walls_x_lim,np.random.rand(1)[0]*self.walls_y_lim])
        if velocity is None:
            self.velocity = np.zeros(2)

        # Start with zero acceleration
        self.acceleration = np.zeros(2)

        # Add to set of all particles
        global counter
        self.id = counter
        counter += 1
        Particle.all += [self]

    def __str__(self) -> str:
        ''' Print statement for particles. '''
        return f"Particle {self.id}at position {self.position} with velocity {self.velocity}"

    def __repr__(self) -> str:
        ''' Debug statement for particles. '''
        return f"{self.__class__.__name__}({self.id},{self.position},{self.velocity})"
    
    def dist(self,other) -> float:
        ''' 
        Calculates squared euclidean distance between particles.
        Usage: my_dist = particle1.dist(particle2).
        '''
        return np.sum((self.position-other.position)**2)
    
    def dirn(self,other) -> float:
        '''
        Calculates the direction unit vector pointing from particle1 to particle2.
        Usage: my_vec = particle1.dist(particle2).
        '''
        return (other.position-self.position)/np.sqrt(self.dist(other))
    
    def normalise_velocity(self, max_speed: float):
        ''' Hard normalise a particle's speed to a specified max speed. '''
        speed = np.sqrt(np.sum(self.velocity)**2)
        if speed > max_speed:
            self.velocity *= max_speed/speed
    


class Prey(Particle):
    '''
    Prey particle for flock of birds simulation.
    '''
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None) -> None:
        super().__init__(position, velocity)


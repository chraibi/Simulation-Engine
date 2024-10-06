# person.py

import numpy as np

class Person:
    # Class to describe each person in simulation
    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        # Check for correct formats

        # Assign defaults
        self.position = position
        self.velocity = velocity

    # Calculate euclidean distance between people
    def dist(self,other)->float:
        return np.sqrt((self.position-other.position)**2)
    
    
    
    
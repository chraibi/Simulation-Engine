import numpy as np
import matplotlib.pyplot as plt
from pedestrian import Person

# Instantiate some people
person1 = Person(np.random.rand(2)*100, np.zeros(2))
person2 = Person(np.random.rand(2)*100, np.zeros(2))
person3 = Person(np.random.rand(2)*100, np.zeros(2))

# Loop through timesteps
time_steps = 100
for t in range(time_steps):
    fig = plt.figure(figsize=[12,12])

    # Loop through each person, update force term 
    for person in Person.all:
        person.update_force_term()
    for person in Person.all:
        person.update_velocity()
        person.position += person.velocity


# TODO: figure out when we move people and when we change velocity, do we move all people first
# Check this works, keep going etc...
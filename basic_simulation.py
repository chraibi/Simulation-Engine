import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pedestrian import Person

# Instantiate some people
person1 = Person(np.random.rand(2)*100, np.zeros(2))
person2 = Person(np.random.rand(2)*100, np.zeros(2))
person3 = Person(np.random.rand(2)*100, np.zeros(2))

# Initialise CSV with current datetime in file name, for uniqueness
now = datetime.datetime.now()
csv_path = "simulation_"+now.date()+"_"+now.time()+".csv"
# Create header columns, 4 for each person id
csv_header = ["time_step"]
for person in Person.all:
    id = str(person.person_id)
    csv_header += ["pos_x_"+id,"pos_y_"+id,"vel_x"+id,"vel_y"+id]
# Write to the CSV path and add header
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)  # Write the header

# Loop through timesteps
time_steps = 100
for t in range(time_steps):
    # Store position and velocity for each person


    # Update force term for each person
    for person in Person.all:
        person.update_force_term()
    # Update velocity and position from force term
    for person in Person.all:
        person.update_velocity()
        person.position += person.velocity # Take a step


# TODO: figure out when we move people and when we change velocity, do we move all people first
# Check this works, keep going etc...
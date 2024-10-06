import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pedestrian import Person
# --------------------------------------------------------------------------------------------------------

# Instantiate some people
person1 = Person(np.random.rand(2)*100, np.zeros(2))
person2 = Person(np.random.rand(2)*100, np.zeros(2))
person3 = Person(np.random.rand(2)*100, np.zeros(2))

# --------------------------------------------------------------------------------------------------------

# Initialise CSV with current datetime in file name, for uniqueness
now = datetime.datetime.now()
csv_path = "simulation_"+str(now.date())+"_"+str(now.time())+".csv"

# Create header columns, 4 for each person id
csv_header = ["time_step"]
for person in Person.all:
    id = str(person.person_id)
    csv_header += ["pos_x_"+id,"pos_y_"+id,"vel_x_"+id,"vel_y_"+id]

# Write to the CSV path and add header
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header) 

# Function to write into CSV with a list, in append mode
def write_row(row: list):
    with open(csv_path, mode='a', newline='') as file:  
        writer = csv.writer(file)
        writer.writerow(row)

# --------------------------------------------------------------------------------------------------------

# Loop through timesteps
time_steps = 100
for t in range(time_steps):
    # Before updating, store position and velocity for each person
    new_csv_row = [t]
    for person in Person.all:
        pos_x, pos_y = person.position[0], person.position[1]
        vel_x, vel_y = person.velocity[0], person.velocity[1]
        new_csv_row += [pos_x, pos_y, vel_x, vel_y]
    write_row(new_csv_row)
    
    # Update force term for each person
    for person in Person.all:
        person.update_force_term()
    # Update velocity and position from force term
    for person in Person.all:
        person.update_velocity()
        person.position += person.velocity # Take a step

# Final state
new_csv_row = [time_steps]
for person in Person.all:
    pos_x, pos_y = person.position[0], person.position[1]
    vel_x, vel_y = person.velocity[0], person.velocity[1]
    new_csv_row += [pos_x, pos_y, vel_x, vel_y]
write_row(new_csv_row)

# --------------------------------------------------------------------------------------------------------


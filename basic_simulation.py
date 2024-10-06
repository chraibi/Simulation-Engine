import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pedestrian import Person


# --------------------------------------------------------------------------------------------------------

# Instantiate some people
num_people = 100
people_instances = []
for i in range(num_people):
    person_instance = Person()
    people_instances.append(person_instance)

# --------------------------------------------------------------------------------------------------------

# Initialise CSV with current datetime in file name, for uniqueness
now = datetime.datetime.now()
csv_path = "Simulation_CSVs/simulation_"+str(num_people)+str(now.date())+"_"+str(now.time())+".csv"

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

# Animate the CSV
fig, ax = plt.subplots()
ax.set_xlim(0, Person.walls_x_lim)  # Set x-axis limits
ax.set_ylim(0, Person.walls_y_lim)  # Set y-axis limits

# Initialise a scatter plot
scat = ax.scatter([], [])

def update(frame):
    # Clear axis between frames, set axes limits
    ax.clear()
    ax.set_xlim(0, Person.walls_x_lim)  # Set x-axis limits
    ax.set_ylim(0, Person.walls_y_lim)  # Set y-axis limits

    # Open row in CSV
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)

        # Loop through the CSV rows until reaching the desired row
        # This must be done since CSV doesn't have indexed data structure
        target_row_index = frame+1 
        for i, row in enumerate(reader):
            if i == target_row_index:
                current_step_strings = row
                current_step = [float(x) for x in current_step_strings] # Convert string -> float!
                break

    # Columns are:
    # time_step, pos_x_0, pos_y_0, vel_x_0, vel_y_0, pos_x_1, pos_x_2, ...
    # Extract x,y positions to scatter
    x_vals = [current_step[i] for i in range(4*num_people+1) if ((i-1)%4 == 0)]
    y_vals = [current_step[i] for i in range(4*num_people+1) if ((i-2)%4 == 0)]

    # Plot scattered points
    ax.scatter(x_vals, y_vals)
    ax.set_title(f"Step {current_step[0]}")

ani = FuncAnimation(fig, update, frames=time_steps, interval=50)

save_as_mp4 = True
if save_as_mp4:
    mp4_path = "Simulation_mp4s/crowd_"+str(num_people)+str(now.date())+"_"+str(now.time())+".MP4"
    ani.save(mp4_path, writer='ffmpeg', fps=30)
    print(f"Saved simulation as mp4 at {mp4_path}.")

plt.show()


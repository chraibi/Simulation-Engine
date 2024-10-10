import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from classes import *

import warnings
warnings.filterwarnings("ignore")

def main():
    # --------------------------------------------------------------------------------------------------------
    # Instantiate some particles

    num_prey = 50
    num_pred = 3
    particle_instances = []
    for i in range(num_prey):
        instance = Prey()
        particle_instances.append(instance)
    for i in range(num_pred):
        instance = Predator()
        particle_instances.append(instance)

    # --------------------------------------------------------------------------------------------------------
    # Create CSV file name

    now = datetime.datetime.now()
    csv_path = "Simulation_CSVs/pred_prey_"+str(num_prey)+"_"+str(num_pred)+"_"+str(now.time())+"_"+str(now.date())+".csv"
    csv_path = csv_path.replace(":","-") # makes file more readable
    Particle.csv_path = csv_path

    # --------------------------------------------------------------------------------------------------------
    # Loop through timesteps
    time_steps = 200
    Particle.num_timesteps = time_steps
    Particle.delta_t = 0.5
    Particle.track_com = False
    Particle.torus = True

    for t in range(time_steps):
        # Print calculation progress
        print(f"----- Computation progress: {t} / {time_steps} -----" ,end="\r", flush=True)

        # Update system
        Particle.timestep_update()

        # Write current system to CSV
        Particle.write_state_to_csv()
    
    # --------------------------------------------------------------------------------------------------------
    # Animate the CSV

    print("-")
    print("\n")

    # Initialise a scatter plot (need all of this)
    fig, ax = plt.subplots(figsize=[7,7])
    fig.canvas.set_window_title(f'Predator Prey animation, {num_prey} prey, {num_pred} predators')
    ax.set_xlim(0, Particle.walls_x_lim)  # Set x-axis limits
    ax.set_ylim(0, Particle.walls_y_lim)  # Set y-axis limits
    scat = ax.scatter([], [])

    # Animate frames by calling update() function
    interval_between_frames = 100 # milliseconds
    ani = FuncAnimation(fig, Particle.animate_timestep, frames=time_steps, \
                        fargs=([ax],), interval=interval_between_frames)

    save_as_mp4 = True
    if save_as_mp4:
        mp4_path = "Simulation_mp4s/pred_prey_"+str(num_prey)+"_"+str(num_pred)+"_"+str(now.time())+"_"+str(now.date())+".MP4"
        mp4_path = mp4_path.replace(":","-")
        fps = 1/(interval_between_frames*(10**(-3))) # period -> frequency
        ani.save(mp4_path, writer='ffmpeg', fps=fps)
        print("\n")
        print(f"Saved simulation as mp4 at {mp4_path}.")

    plt.show()


    # Generate list of instances for each desired class
    # Compute datetime and file names
    # Update Particle.num_timesteps
    # for timestep in range(num_steps):
    #       nice print statement progress
    #       current_time = timestep*Particle.delta_t
    #       Particle.timestep_update()
    #       Particle.write_to_csv(filename)
    # print computing done, starting animation
    # fig, ax = plt.figure
    # ani = FuncAnimation( Particle.animation_timestep, fargs=(ax)   )


if __name__=="__main__":
    # Ask for sys.input from user, enter for defaults
    # 1 for birds, 2 for galaxy, 3 for crowd
    # Torus or no
    # How many birds (int)
    # How many timesteps, enter for default (100)
    # Save as mp4 ?
    # main(type, num_particles, save) etc
    main()
    
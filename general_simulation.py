import datetime
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from classes import *
from simulation_classes import *

import warnings
warnings.filterwarnings("ignore")

def main(args):
     # Unpack arguments
    
    type = args.type
    time_steps = args.steps
    num1 = args.num1
    num2 = args.num2
    save_as_mp4 = args.save_mp4
    user_mp4_path = args.mp4_path

    # --------------------------------------------------------------------------------------------------------
    # Instantiate some particles

    now = datetime.datetime.now()
    show_graph = False # secondary axis

    if type == 'birds':
        Environment.background_type = "sky"
        num_prey = num1
        num_pred = num2
        for i in range(num_prey):
            Prey()
        for i in range(num_pred):
            Predator()
        Particle.track_com = False
        Particle.torus = True
        csv_path = f"Simulation_CSVs/{type}_{str(num_prey)}_{str(num_pred)}_{str(now.time())}_{str(now.date())}.csv"
        mp4_path = f"Simulation_mp4s/{type}_{str(num_prey)}_{str(num_pred)}_{str(now.time())}_{str(now.date())}.MP4"
        window_title = f'Predator Prey animation, {num_prey} prey, {num_pred} predators'

    elif type == 'nbody':
        Environment.background_type = "space"
        Particle.walls_x_lim = 1000
        Particle.walls_y_lim = 1000
        num_bodies = num1
        for i in range(num_bodies):
            Star()
        Particle.track_com = True
        Particle.torus = False
        csv_path = f"Simulation_CSVs/{type}_{str(num_bodies)}_{str(now.time())}_{str(now.date())}.csv"
        mp4_path = f"Simulation_mp4s/{type}_{str(num_bodies)}_{str(now.time())}_{str(now.date())}.MP4"
        window_title = f'N-body animation, {num_bodies} bodies'

    elif type == 'evac':
        Environment.background_type = "room"
        num_people = num1
        Particle.num_evacuees = num_people
        Particle.walls_x_lim = 10
        Particle.walls_y_lim = 10
        classroom = True
        if classroom:
            Particle.walls_x_lim = 15
            Particle.walls_y_lim = 10
            # Target
            Target(np.array([Particle.walls_x_lim+1,2.5]))
            # 3 walls with door
            Wall(np.array([0,0]),np.array([0,Particle.walls_y_lim]))
            Wall(np.array([0,0]),np.array([Particle.walls_x_lim, 0]))
            Wall(np.array([0,Particle.walls_y_lim]),np.array([Particle.walls_x_lim, Particle.walls_y_lim]))
            # Wall with door opening
            Wall(np.array([Particle.walls_x_lim, 0]),np.array([Particle.walls_x_lim, 2]))
            Wall(np.array([Particle.walls_x_lim, 3]),np.array([Particle.walls_x_lim, Particle.walls_y_lim]))
            # Teacher desk
            Wall(np.array([Particle.walls_x_lim-1, 3]),np.array([Particle.walls_x_lim-1, 7]))


        first = False
        if first:
            Wall(np.array([0,0]),np.array([0,Particle.walls_y_lim]))
            Wall(np.array([0,0]),np.array([Particle.walls_x_lim, 0]))
            Wall(np.array([0,Particle.walls_y_lim]),np.array([Particle.walls_x_lim, Particle.walls_y_lim]))
            Wall(np.array([Particle.walls_x_lim, 0]),np.array([Particle.walls_x_lim-1, 4.5]))
            Wall(np.array([Particle.walls_x_lim-1, 5.5]),np.array([Particle.walls_x_lim, Particle.walls_y_lim]))
            Wall(np.array([3,5]),np.array([8, 5]))

        show_graph = True
        for i in range(num_people):
            Human()
        Particle.track_com = False
        Particle.torus = False
        csv_path = f"Simulation_CSVs/{type}_{str(num_people)}_{str(now.time())}_{str(now.date())}.csv"
        mp4_path = f"Simulation_mp4s/{type}_{str(num_people)}_{str(now.time())}_{str(now.date())}.MP4"
        window_title = f'Evacuation simulation, {num_people} people'
        
    # --------------------------------------------------------------------------------------------------------
    # Create CSV file name

    csv_path = csv_path.replace(":","-") # makes file more readable
    Particle.csv_path = csv_path

    # --------------------------------------------------------------------------------------------------------
    # Loop through timesteps

    Particle.num_timesteps = time_steps
    Particle.delta_t = 0.1
    
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
    if show_graph:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 7))
        scat = ax2.scatter([], [])
    else:
        fig, ax = plt.subplots(figsize=[7,7])
    fig.canvas.set_window_title(window_title)
    ax.set_xlim(-1, Particle.walls_x_lim+1)  # Set x-axis limits
    ax.set_ylim(-1, Particle.walls_y_lim+1)  # Set y-axis limits
    scat = ax.scatter([], [])
    


    # Animate frames by calling update() function
    interval_between_frames = 100 # milliseconds
    if show_graph:
        ani = FuncAnimation(fig, Particle.animate_timestep, frames=time_steps, \
                        fargs=([ax],[ax2]), interval=interval_between_frames)
    else:
        ani = FuncAnimation(fig, Particle.animate_timestep, frames=time_steps, \
                        fargs=([ax],), interval=interval_between_frames)

    save_as_mp4 = True
    if save_as_mp4:
        mp4_path = mp4_path.replace(":","-")
        if user_mp4_path is not None:
            mp4_path = user_mp4_path
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
    # Create the argument parser
    parser = argparse.ArgumentParser(description="General Simulation Engine input options.")
    
    # Add arguments
    parser.add_argument('--type', type=str, help='The type of simulation [evac, birds, nbody]')
    parser.add_argument('--steps', type=int, help='The number of timesteps in the simulation [10 <= N <~ 500, default 100]', default=100)
    parser.add_argument('--num1', type=int, help='The number of particles in the simulation [1<= N <~ 500, default 20]', default=20)
    parser.add_argument('--num2', type=int, help='The number of secondary particles in the simulation (eg Predators) [1<= N <~ 500, default 3]', default=3)
    parser.add_argument('--save_mp4', type=bool, help='Whether to save the simulation as an mp4 video [True, False, default True]', default=True)
    parser.add_argument('--mp4_path', type=str, help="(Optional) The mp4's relative path string (will create within current directory).", default=None)

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)
    
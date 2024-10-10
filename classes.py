# classes.py - generalising pedestrian.py

import os
import csv
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

class Particle:
    '''
    Parent class for particles in a 2D plane
    '''
    # -------------------------------------------------------------------------
    # Attributes

    #  Dictionaries for population count and maximum ID number of each child class
    pop_counts_dict = {} 
    max_ids_dict = {}

    # Dictionary of all child classes, referenced by ID number
    # eg {bird: {0:instance0, 1:instance1, ...}, plane: {0: ...}, ... }
    # This is used to fully encode the system's state at each timestep.
    all = {}

    # Track time and time step
    delta_t = 0.01
    current_time = 0
    current_step: int = 0
    num_timesteps: int = 100

    # Basic wall boundaries (Region is [0,walls_x_lim]X[0,walls_y_lim] )
    walls_x_lim: float = 100
    walls_y_lim: float = 100

    # Bool whether to track COM when plotting
    track_com: bool = True
    
    # Initialisation function
    def __init__(self,
                position: np.ndarray = None,
                velocity: np.ndarray = None) -> None:
        '''
        Initiate generic particle, with optional position and velocity inputs.
        Start with random position, and zero velocity and zero acceleration.
        Set an ID value and then increment dictionaries
        '''
        # ---------------
        # Motion

        # If no starting position given, assign random within 2D wall limits
        if position is None:
            self.position = np.array([np.random.rand(1)[0]*self.walls_x_lim,np.random.rand(1)[0]*self.walls_y_lim])
        # If no starting velocity given, set it to zero.
        if velocity is None:
            self.velocity = np.zeros(2)

        # Extrapolate last position from starting position and velocity
        if velocity is None:
            self.last_position = self.position
        else:
            # v = (current-last)/dt , so last = current - v*dt
            self.last_position = self.position - self.velocity*self.delta_t

        # Initialise acceleration as attribute
        self.acceleration = np.zeros(2)

        # --------------
        # Indexing

        # Get Child class name of current instance
        class_name = self.__class__.__name__
        
        # Population count
        if class_name not in Particle.pop_counts_dict:
            Particle.pop_counts_dict[class_name] = 0
        Particle.pop_counts_dict[class_name] += 1

        # ID - index starts at 0
        if class_name not in Particle.max_ids_dict:
            Particle.max_ids_dict[class_name] = 0
        else:
            Particle.max_ids_dict[class_name] += 1
        self.id = Particle.max_ids_dict[class_name]

        # Add instance to 'all' dict
        if class_name not in Particle.all:
            Particle.all[class_name] = {}
        Particle.all[class_name][self.id] = self

    # -------------------------------------------------------------------------
    # Instance management utilities
    # TODO: Make some of these hidden!

    @classmethod
    def get_count(cls):
        ''' Return a class type count. eg  num_birds = Bird.get_count(). '''
        return Particle.pop_counts_dict.get(cls.__name__, 0)
    @classmethod
    def get_max_id(cls):
        ''' Return a class max id. eg max_id_birds = Bird.get_max_id(). '''
        return Particle.max_ids_dict.get(cls.__name__, 0)
    
    @classmethod
    def remove_by_id(cls, id):
        ''' Remove class instance from list of instances by its id. '''
        if id in Particle.all[cls.__name__]:
            del Particle.all[cls.__name__][id]
            Particle.pop_counts_dict[cls.__name__] -= 1
        else:
            pass

    @classmethod
    def get_instance_by_id(cls, id):
        ''' Get class instance by its id. If id doesn't exist, throw a KeyError.'''
        if id in Particle.all[cls.__name__]:
            return Particle.all[cls.__name__][id]
        else:
            raise KeyError(f"Instance with id {id} not found in {cls.__name__}.")
        
    @classmethod
    def iterate_class_instances(cls):
        ''' Iterate over all instances of a given class by id. '''
        # This function is a 'generator' object in Python due to the use of 'yield'.
        # It unpacks each {id: instance} dictionary item within our Particle.all[classname] dictionary
        # It then 'yields' the instance. Can be used in a for loop as iterator.
        for id, instance in Particle.all.get(cls.__name__, {}).items():
            yield instance

    @staticmethod
    def iterate_all_instances():
        ''' Iterate over all existing child instances. '''
        # Create dictionary with all child instances
        dict_list = {}
        for i in Particle.all.values():
            dict_list.update(i)
        # Create generator through the dictionary values (instances)
        for id, instance in dict_list.items():
            yield instance
        
    def __str__(self) -> str:
        ''' Print statement for particles. '''
        return f"Particle {self.id} at position {self.position} with velocity {self.velocity}."

    def __repr__(self) -> str:
        ''' Debug statement for particles. '''
        return f"{self.__class__.__name__}({self.id},{self.position},{self.velocity})"

    # -------------------------------------------------------------------------
    # Distance utilities

    torus = False
    '''
    Periodic boundaries -> We have to check different directions for shortest dist.
    Need to check tic-tac-toe grid of possible directions:
            x | x | x
            ---------
            x | o | x
            ---------
            x | x | x
    We work from top right, going clockwise.
    '''
    up, right = np.array([0,walls_y_lim]), np.array([walls_x_lim,0])
    torus_offsets = [np.zeros(2), up+right, right, -up+right, -up, -up-right, -right, up-right, up]

    def torus_dist(self,other):
        directions = [(other.position + i) - self.position  for i in Particle.torus_offsets]
        distances = [np.sum(i**2) for i in directions]
        mindex = np.argmin(distances)
        return distances[mindex], directions[mindex]

    def dist(self,other, return_both: bool = False):
        ''' 
        Calculates SQUARED euclidean distance between particles.
        Usage: my_dist = particle1.dist(particle2).
        If Particle.torus, then finds shortest squared distance from set of paths.
        '''
        if Particle.torus:
            dist, dirn = self.torus_dist(other)
        else:
            dirn = other.position - self.position
            dist = np.sum((dirn)**2)

        if return_both:
            return dist, dirn
        else:
            return dist
            
    def unit_dirn(self,other) -> float:
        '''
        Calculates the direction unit vector pointing from particle1 to particle2.
        Usage: my_vec = particle1.dirn(particle2).
        '''        
        dist, dirn = self.dist(other,return_both=True)
        return dirn/np.sqrt(dist)
               
    def enforce_speed_limit(self):
        ''' Hardcode normalise a particle's velocity to a specified max speed. '''
        # Hardcode speed limit, restrict displacement
        speed = np.sqrt(np.sum(self.velocity**2))
        if speed > self.max_speed:
            # Change velocity
            self.velocity *= self.max_speed/speed
            # Change current position to backtrack
            self.position = self.last_position + self.velocity*Particle.delta_t

    def torus_wrap(self):
        ''' Wrap coordinates into Torus world with modulo functions'''
        x,y = self.position
        x = x % Particle.walls_x_lim
        y = y % Particle.walls_y_lim
        self.position = np.array([x,y])

    @classmethod
    def centre_of_mass_class(cls):
        ''' Compute COM of all particle instances. '''
        total_mass = 0
        com = np.zeros(2)
        # Call generator to run over all particle instances
        for instance in cls.iterate_class_instances():
            mass = instance.mass
            com += mass*instance.position
            total_mass += mass
        com *= 1/total_mass
        return com
    
    @staticmethod
    def centre_of_mass():
        ''' Compute COM of all particle instances. '''
        total_mass = 0
        com = np.zeros(2)
        # Call generator to run over all particle instances
        for instance in Particle.iterate_all_instances():
            mass = instance.mass
            com += mass*instance.position
            total_mass += mass
        com *= 1/total_mass
        return com

    @staticmethod
    def scene_scale():
        ''' Compute the maximum x or y distance a particle has from the COM. '''
        com = Particle.centre_of_mass()
        max_dist = 0
        # Call generator to find max dist from COM
        for instance in Particle.iterate_all_instances():
            vec_from_com = instance.position - com
            for i in vec_from_com:
                if i > max_dist:
                    max_dist = i
                else:
                    pass
        return max_dist
    
    def orient_to_com(self, com, scale):
        ''' Affine translation on point coordinates to prepare for plotting.  '''
        centre = np.array([0.5*Particle.walls_x_lim, 0.5*Particle.walls_y_lim])
        term = np.min(centre)
        return centre + (self.position - com) * (term/scale)

    # -------------------------------------------------------------------------
    # Main timestep function

    @staticmethod
    def timestep_update():
        '''
        Main timestep function. 
        - Calls each child class instance to update its acceleration,
            according to its own force rules. 
        - Uses 'Verlet Integration' timestepping method, predicting instance's position after a 
            timestep using its current position, last position, and acceleration:
            x_next = 2*x_now - x_last + acc*(dt)^2
        - Passes predicted new position through checks, including speed limits,
            and torus modulo function on coordinates.
        '''
        for i in Particle.iterate_all_instances():
            # Let particle update its acceleration 
            i.update_acceleration()

            # Verlet Integration
            # Use tuple unpacking so we dont need a temp variable
            i.position, i.last_position = (2*i.position - i.last_position + \
                                            i.acceleration*(Particle.delta_t)**2), i.position
            
            # Update velocity
            displacement = (i.position - i.last_position)
            i.velocity = displacement/Particle.delta_t

            # Enforce speed limit
            if i.max_speed is not None:
                i.enforce_speed_limit()

            # Enforce torus wrapping
            if Particle.torus:
                i.torus_wrap()
        
        # Increment time
        Particle.current_time += Particle.delta_t
        Particle.current_step += 1
    
    # -------------------------------------------------------------------------
    # CSV utilities

    # CSV path, to be set by main script with datetime for reference
    csv_path = "my_csv.csv"

    @staticmethod
    def write_state_to_csv():
        '''
        Takes Particle system state at the current time, and compresses into CSV.
        Iterates through each class, and within that each class instance.
        Calls each class's own method to write its own section.
        '''
        #--------------------------------
        # Compose CSV row entry
        system_state_list = [Particle.current_step, Particle.current_time]

        # Iterate through all current child classes
        for classname in Particle.pop_counts_dict.keys():

            # Get class by string name
            my_class = globals()[classname]

            # Initialise class specific list
            class_list = [classname, Particle.pop_counts_dict[classname]]

            # Iterate through all instances
            for child in my_class.iterate_class_instances():
                # Add instance info to list using its write_csv_list function
                class_list += child.write_csv_list()

            # Add child class info to main list
            class_list += ['|']
            system_state_list += class_list

        # End CSV row with 'END'
        system_state_list += ['END']

        # ------------------------------------
        # Writing entry to file

        # If CSV doesn't exist, make it with an initial header on row 0, then write state
        if not os.path.exists(Particle.csv_path):
            with open(Particle.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                header_row = ['Timestep', 'Time', 'ClassName', 'ClassPop', 'InstanceID', 'Attributes', '...','|','ClassName','...','|','END']
                writer.writerow(header_row)
                writer.writerow(system_state_list)
        # Else open in append mode and write
        else:
            with open(Particle.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(system_state_list)

    @staticmethod
    def load_state_from_csv(timestep):
        '''
        Reads from a CSV containing the compressed Particle system state at a specific time.
        Iterates through each class, and within that each class instance.
        Parses to decompress the format outlined in write_state_to_csv.
        '''
        # ------------------------------------
        # Read row from CSV

        with open(Particle.csv_path, mode='r', newline='') as file:
            # Loop through the CSV rows until reaching the desired row
            # (This must be done since CSV doesn't have indexed data structure)
            reader = csv.reader(file)
            target_row_index = timestep+1 
            for i, row in enumerate(reader):
                if i == target_row_index:
                    system_state_list = row.copy()
                    break
        
        # ------------------------------------
        # Parse row into a full Particle system state

        # Parse timestep info, shift index
        Particle.current_step, Particle.current_time = system_state_list[0], system_state_list[1]
        idx_shift = 2

        # Loop through blocks for each child class
        while True:
            # Check if reached the end of row
            if system_state_list[idx_shift] == 'END':
                break

            # Parse class and number of instances, shift index
            my_class = globals()[system_state_list[idx_shift]]
            class_pop = int(system_state_list[idx_shift+1])
            idx_shift += 2

            # Get rid of all existing instances of that class
            max_id = Particle.max_ids_dict[my_class.__name__]
            for id in range(max_id+1):
                my_class.remove_by_id(id)
            
            # Loop through each instance in csv row
            for i in range(class_pop):
                # Create new child instance
                child = my_class()

                # Assign attributes by reading the system_state_list for that class
                # This calls to child class's method to read each instance
                idx_shift = child.read_csv_list(system_state_list, idx_shift)

                # Add child to current 'all' list
                Particle.all[my_class.__name__][child.id] = child
                
            # Check for pipe | at the end, then move past it
            if system_state_list[idx_shift] != '|':
                raise IndexError(f"Something wrong with parsing, ~ column {idx_shift}.")
            idx_shift += 1
        
    # -------------------------------------------------------------------------
    # Animation utilities

    @staticmethod
    def animate_timestep(timestep, ax):
        '''
        Draws the state of the current system onto a matplotlib ax object provided.
        This function will be called by FuncAnimation at each timestep in the main simulation script.
        Calls upon each child instance to plot itself, 
        as well as calling the Environment class for backdrop.
        ''' 
        # Unpack wrapped ax object
        ax = ax[0]

        # Print calculation progress
        print(f"----- Animation progress: {timestep} / {Particle.num_timesteps} -----" ,end="\r", flush=True)

        # Clear axis between frames, set axes limits again and title
        ax.clear()
        ax.set_xlim(0, Particle.walls_x_lim)  # Set x-axis limits
        ax.set_ylim(0, Particle.walls_y_lim)  # Set y-axis limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Time step: {Particle.current_step}, Time: {Particle.current_time}.")

        # Call upon Environment class to draw the frame's backdrop
        Environment.draw_backdrop(ax)

        # Load in system state from CSV
        Particle.load_state_from_csv(timestep)

        # Decide if tracking the COM in each frame
        if Particle.track_com:
            com = Particle.centre_of_mass()
            scene_scale = Particle.scene_scale()
        else:
            com, scene_scale = None, None

        # Iterate over child instances in system and plot
        for instance in Particle.iterate_all_instances():
            instance.instance_plot(ax,com,scene_scale)

class Prey(Particle):
    '''
    Prey particle for flock of birds simulation.
    '''
    # -------------------------------------------------------------------------
    # Attributes

    prey_dist_thresh = 10**2
    prey_repulsion_force = 5
    com_attraction_force = 10
    random_force = 2
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None) -> None:
        '''
        Initialises a Prey bird, inheriting from the Particle class.
        '''
        super().__init__(position, velocity)

        # Prey specific attributes
        self.mass = 5
        self.max_speed = 10

    # -------------------------------------------------------------------------
    # Main force model 

    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''
        # Instantiate force term
        force_term = np.zeros(2)

        # Prey repulsion force - currently scales with 1/d
        for bird in Prey.iterate_class_instances():
            if bird == self:
                continue
            elif self.dist(bird) < Prey.prey_dist_thresh:
                force_term += self.unit_dirn(bird)*(self.prey_repulsion_force/(np.sqrt(self.dist(bird))))
            else:
                continue
    
        # Attraction to COM
        com = Prey.centre_of_mass_class()
        attract_dist = np.sum((com - self.position)**2)
        force_term += (com - self.position)*(self.com_attraction_force/(attract_dist))

        # Random force - stochastic noise
        # Generate between [0,1], map to [0,2] then shift to [-1,1]
        force_term += ((np.random.rand(2)*2)-1)*self.random_force

        # Update acceleration = Force / mass
        self.acceleration = force_term / self.mass
    
    # -------------------------------------------------------------------------
    # CSV utilities

    def write_csv_list(self):
        '''
        Format for compressing each Prey instance into CSV.
        '''
        # Individual child instance info
        return [self.id, \
                self.position[0], self.position[1], \
                self.last_position[0],self.last_position[1],
                self.velocity[0], self.velocity[1],
                self.acceleration[0], self.acceleration[1] ]

    def read_csv_list(self, system_state_list, idx_shift):
        '''
        Format for parsing the compressed Prey instances from CSV.
        '''
        self.id = system_state_list[idx_shift]
        self.position = np.array([float(system_state_list[idx_shift+1]), \
                                    float(system_state_list[idx_shift+2])])
        self.last_position = np.array([float(system_state_list[idx_shift+3]), \
                                    float(system_state_list[idx_shift+4])])
        self.velocity = np.array([float(system_state_list[idx_shift+5]), \
                                    float(system_state_list[idx_shift+6])])
        self.acceleration = np.array([float(system_state_list[idx_shift+7]), \
                                    float(system_state_list[idx_shift+8])])
        # Update idx shift to next id and return
        return idx_shift+9

    # -------------------------------------------------------------------------
    # Animation utilities

    @staticmethod
    def create_irregular_triangle(angle_rad):
        '''
        Create irregular triangle marker for plotting instances.
        '''
        # Define vertices for an irregular triangle (relative to origin)
        triangle = np.array([[-0.5, -1], [0.5, -1], [0.0, 1]])
        # Create a rotation matrix
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad),  np.cos(angle_rad)]])
        # Apply the rotation to the triangle vertices
        return triangle @ rotation_matrix.T

    def instance_plot(self, ax, com=None, scale=None):
        ''' 
        Plots individual Prey particle onto existing axis. 
        '''

        # Get plot position in frame
        plot_position = self.position
        if (com is not None) and (scale is not None):
            plot_position = self.orient_to_com(com, scale)

        # Get direction angle from velocity
        theta = np.arctan2(self.velocity[1], self.velocity[0]) - np.pi/2

        # Create a Polygon patch to represent the irregular triangle
        triangle_shape = Prey.create_irregular_triangle(theta)
        polygon = Polygon(triangle_shape, closed=True, facecolor='white', edgecolor='black')
        
        # Create and apply transformation of the polygon to the point
        t = Affine2D().translate(plot_position[0], plot_position[1]) + ax.transData
        polygon.set_transform(t)

        # Plot polygon
        ax.add_patch(polygon)
        









class Environment:
    '''
    Class containing details about the simulation environment, walls etc
    '''
    # Background colour for each type of environment
    background_type = 'sky'
    background_colour_dict = {"sky": "skyblue",
                              "space": "k",
                              "room": "w"}
    
    @staticmethod
    def draw_background_colour(ax):
        ax.set_facecolor(Environment.background_colour_dict[Environment.background_type])

    @staticmethod
    def draw_objects(ax):
        pass

    @staticmethod
    def draw_backdrop(ax):
        '''
        Called by Particle.animate_timestep to set background for each frame, 
         before drawing its particle objects over the top.
        An ax is passed in and we call different functions to draw environment elements
        '''
        Environment.draw_background_colour(ax)
        Environment.draw_objects(ax)

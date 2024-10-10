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

    @staticmethod
    def centre_of_mass():
        ''' Compute COM of all particle instances. '''
        total_mass = 0
        com = np.zeros(2)
        # Call generator to run over all particle instances
        for instance in Particle.iterate_all_instances:
            mass = instance.mass
            com += mass*instance.position
            total_mass += mass
        com *= 1/total_mass

    @staticmethod
    def scene_scale():
        ''' Compute the maximum x or y distance a particle has from the COM. '''
        com = Particle.centre_of_mass()
        max_dist = 0
        # Call generator to find max dist from COM
        for instance in Particle.iterate_all_instances:
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
        return centre + (self.position - com)*term/scale

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
        for i in Particle.iterate_all_instances:
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
    # TODO: Make some of these hidden!

    # CSV path, to be set by main script
    csv_path = "my_csv.csv"

    @staticmethod
    def write_state_to_csv():
        '''
        Takes current system state of all child instances at a particular timestep.
        Composes a CSV row entry to encode this, calling each child class to write it's own instances.
        Used to recursively record system's state for each timestamp in SSD.
        '''
        # Compose CSV row entry
        system_state_list = [Particle.current_step, Particle.current_time]
        for classname in Particle.pop_counts_dict.keys():
            # Access child class by string name using globals() dictionary
            my_class = globals()[classname]
            # Call child class's list writer, add to general list
            system_state_list += my_class.write_csv_list()
        system_state_list += ['END']

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
        Reads from a CSV of system states. Iterates through rows until it reaches right timestep.
        Then calls on each child class to read it's entries when its name is mentioned, 
        each time shifting a starting index.
        '''

        # Open correct row in CSV
        with open(Particle.csv_path, mode='r', newline='') as file:
            # Loop through the CSV rows until reaching the desired row
            # This must be done since CSV doesn't have indexed data structure
            reader = csv.reader(file)
            target_row_index = timestep+1 
            for i, row in enumerate(reader):
                if i == target_row_index:
                    system_state_list = row.copy()
                    #current_step = [float(x) for x in current_step_strings] # Convert string -> float!
                    break
        
        # Parse timestep info
        Particle.current_step, Particle.current_time = system_state_list[0], system_state_list[1]
        idx_shift = 2
        while True:
            if system_state_list[idx_shift] == 'END':
                break
            class_name = globals()[system_state_list[idx_shift]]
            # Call child class's list reader, get new start point
            # This reinstantiaties all instances under the hood
            idx_shift = class_name.read_csv_list(system_state_list, idx_shift)
        
    # -------------------------------------------------------------------------
    # Animation utilities
    # TODO: Make some of these hidden!

    @staticmethod
    def animate_timestep(timestep, ax):
        '''
        Draws the state of the current system onto a matplotlib ax object provided.
        This function will be called by FuncAnimation at each timestep in the main simulation script.
        Calls upon each child instance to plot itself, 
        as well as calling the Environment class for backdrop.
        ''' 
        # Print calculation progress
        print(f"----- Animation progress: {timestep} / {Particle.num_timesteps} -----" ,end="\r", flush=True)

        # Clear axis between frames, set axes limits again and title
        ax.clear()
        ax.set_xlim(0, Particle.walls_x_lim)  # Set x-axis limits
        ax.set_ylim(0, Particle.walls_y_lim)  # Set y-axis limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Time step: {round((Particle.current_step))}, Time: {Particle.current_time}.")

        # Call upon Environment class to draw the frame's backdrop
        Environment.draw_backdrop(ax)

        # Load in system state from CSV
        Particle.load_state_from_csv(timestep)

        # Decide if tracking the COM in each frame
        if Particle.track_com:
            com = Particle.centre_of_mass()
            scene_scale = Particle.scene_scale
        else:
            com, scene_scale = None, None

        # Iterate over child instances in system and plot
        for instance in Particle.iterate_all_instances:
            instance.instance_plot(ax,com,scene_scale)

class Prey(Particle):
    '''
    Prey particle for flock of birds simulation.
    '''
    max_speed = 5

    mass = 7 

    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None) -> None:
        super().__init__(position, velocity)
        pass

    def update_acceleration(self):
        # go through all in Particle.all[Prey] and Particle.all[Predator] to work out forces
        # acceleration by dividing by self.mass
        # Use this point to track killing as well: use remove_by_id method if any Predator too close
        # Could have rule that this spawns in a new predator?? cool
        pass
    
    # -------------------------------------------------------------------------
    # CSV utilities

    @classmethod
    def write_csv_list(cls):
        # Formats current set of class instances into a list, to be added to a CSV row
        # by the main CSV function
        # Prey, NumPrey, ID1, posx, posy, velx, vely, .. ID2, ...,  ,|,
        # Converts this into NumPrey many instances to recover state from CSV
        child_list = [cls.__name__, Particle.pop_counts_dict[cls.__name__]]
        for child in cls.iterate_class_instances():
            # Individual child instance
            child_list += [child.id, \
                           child.position[0], child.position[1], \
                           child.last_position[0],child.last_position[1],
                           child.velocity[0], child.velocity[1],
                           child.acceleration[0], child.acceleration[1]
                           ]
        # End pipe for parsing
        child_list += ['|']
        return child_list

    @classmethod
    def read_csv_list(cls, system_state_list: list, idx_shift: int):
        # Given a list from main CSV reading function. This looks like:
        # Prey, NumPrey, ID1, posx, posy, velx, vely, .. ID2, ...,  ,|,
        # Converts this into NumPrey many instances to recover state from CSV

        # First get rid of all existing instances
        max_id = Particle.max_ids_dict[cls.__name__]
        for id in range(max_id+1):
            cls.remove_by_id(id)
        
        # Get number of children, then shift to start of parsing block
        class_pop = int(system_state_list[idx_shift+1])
        idx_shift += 2
        # Loop through each child instance
        for i in range(class_pop):
            # Create new child instance, assign attributes
            child = cls()
            child.id = system_state_list[idx_shift]
            child.position = np.array([float(system_state_list[idx_shift+1]), \
                                       float(system_state_list[idx_shift+2])])
            child.last_position = np.array([float(system_state_list[idx_shift+3]), \
                                       float(system_state_list[idx_shift+4])])
            child.velocity = np.array([float(system_state_list[idx_shift+5]), \
                                       float(system_state_list[idx_shift+6])])
            child.acceleration = np.array([float(system_state_list[idx_shift+7]), \
                                       float(system_state_list[idx_shift+8])])
            
            # Add child to 'all' list
            Particle.all[cls.__name__][child.id] = child
            # Update idx shift to next child id
            idx_shift += 9

        # Check for correct parsing at the end
        if system_state_list[idx_shift] != '|':
            raise IndexError(f"Something wrong with parsing, ~ column {idx_shift}.")
        
        # Return shifted index, with all children now instantiated.
        return idx_shift

    # -------------------------------------------------------------------------
    # Animation utilities

    # Triangle creator for directed markers
    @staticmethod
    def create_irregular_triangle(angle_rad):
        # Define vertices for an irregular triangle (relative to origin)
        triangle = np.array([[-0.5, -1], [0.5, -1], [0.0, 1]])
        # Create a rotation matrix
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad),  np.cos(angle_rad)]])
        # Apply the rotation to the triangle vertices
        return triangle @ rotation_matrix.T

    def instance_plot(self, ax, com=None, scale=None):
        ''' Plots individual Prey particle onto existing axis. '''

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

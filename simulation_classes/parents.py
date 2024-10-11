import os
import csv
import numpy as np

class Particle:
    '''
    Parent class for particles in a 2D plane
    '''
    # -------------------------------------------------------------------------
    # Attributes

    #  Dictionaries for population count and maximum ID number of each child class
    pop_counts_dict = {} 
    max_ids_dict = {}

    # Dictionary for tracking kills for each timestep
    kill_count = 0
    kill_record = {0:0}
    num_evacuees = 0 

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
        self.alive=1
        # ---------------
        # Motion
        self.max_speed = None

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
    def get_instance_by_id(cls, id):
        ''' Get class instance by its id. If id doesn't exist, throw a KeyError.'''
        if id in Particle.all[cls.__name__]:
            return Particle.all[cls.__name__][id]
        else:
            raise KeyError(f"Instance with id {id} not found in {cls.__name__}.")
        
    def unalive(self):
        ''' 
        Sets the class instance with this id to be not alive, decrements the class count.
        '''
        self.alive=0
        Particle.pop_counts_dict[self.__class__.__name__] -= 1
        Particle.kill_count += 1
        Particle.kill_record[Particle.current_step] = Particle.kill_count

    @classmethod
    def iterate_class_instances(cls):
        ''' Iterate over all instances of a given class by id. '''
        # This function is a 'generator' object in Python due to the use of 'yield'.
        # It unpacks each {id: instance} dictionary item within our Particle.all[classname] dictionary
        # It then 'yields' the instance. Can be used in a for loop as iterator.
        for id, instance in Particle.all.get(cls.__name__, {}).items():
            if instance.alive == 1:
                yield instance

    @staticmethod
    def iterate_all_instances():
        ''' Iterate over all existing child instances. '''
        # Create big flattened dictionary with all child instances
        dict_list = {}
        for i in Particle.all.values():
            dict_list.update(i)
        # Create generator through the dictionary values (instances)
        for id, instance in dict_list.items():
            if instance.alive == 1:
                yield instance
        
    def __str__(self) -> str:
        ''' Print statement for particles. '''
        if self.alive==1:
            return f"Particle {self.id} at position {self.position} with velocity {self.velocity}."
        else:
            return f"Dead Particle {self.id} at position {self.position} with velocity {self.velocity}."

    def __repr__(self) -> str:
        ''' Debug statement for particles. '''
        if self.alive==1:
            return f"{self.__class__.__name__}({self.id},{self.position},{self.velocity})"
        else:
            return f"dead_{self.__class__.__name__}({self.id},{self.position},{self.velocity})"

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
            flag = i.update_acceleration()
            if flag==1:
                # i has been killed
                continue

            # Find last position from velocity - avoids torus wrapping problems
            i.last_position = i.position - i.velocity*Particle.delta_t

            # Verlet Integration
            # Use tuple unpacking so we dont need a temp variable
            i.position, i.last_position = 2*i.position - i.last_position +i.acceleration*((Particle.delta_t)**2), i.position
            
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

        # Update kill records
        Particle.kill_record[Particle.current_step] = Particle.kill_count
    
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
            #for id, instance in Particle.all.get(my_class.__name__, {}).items():
            #    instance.unalive()
            Particle.pop_counts_dict[my_class.__name__] = 0
            Particle.max_ids_dict[my_class.__name__] = -1
            Particle.all[my_class.__name__] = {}

            # Loop through each instance in csv row
            for i in range(class_pop):
                # Create new child instance
                child = my_class()

                # Assign attributes by reading the system_state_list for that class
                # This calls to child class's method to read each instance
                idx_shift = child.read_csv_list(system_state_list, idx_shift)
                
            # Check for pipe | at the end, then move past it
            if system_state_list[idx_shift] != '|':
                raise IndexError(f"Something wrong with parsing, ~ column {idx_shift}.")
            idx_shift += 1
        
    # -------------------------------------------------------------------------
    # Animation utilities

    @staticmethod
    def animate_timestep(timestep, ax, ax2=None):
        '''
        Draws the state of the current system onto a matplotlib ax object provided.
        This function will be called by FuncAnimation at each timestep in the main simulation script.
        Calls upon each child instance to plot itself, 
        as well as calling the Environment class for backdrop.
        ''' 
        # Unpack wrapped ax object
        ax = ax[0]
        if ax2 is not None:
            ax2 = ax2[0]
        
        # Print calculation progress
        print(f"----- Animation progress: {timestep} / {Particle.num_timesteps} -----" ,end="\r", flush=True)

        # Clear axis between frames, set axes limits again and title
        ax.clear()
        ax.set_xlim(-1, Particle.walls_x_lim+1)  # Set x-axis limits
        ax.set_ylim(-1, Particle.walls_y_lim+1)  # Set y-axis limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Time step: {Particle.current_step}, Time: {round(float(Particle.current_time),2)}.")

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

        # Plot second graph
        ax2.clear()
        ax2.set_xlim(0, Particle.num_timesteps)  # Set x-axis limits
        ax2.set_ylim(0, Particle.num_evacuees)  # FIX THIS
        ax2.set_title(f"Evacuated over time")
        t_vals = []
        y_vals = []
        for key, item in Particle.kill_record.items():
            if key <= timestep:
                t_vals += [key]
                y_vals += [item]
        ax2.plot(t_vals, y_vals, c='b')
        ax2.scatter(timestep,Particle.kill_record[timestep], marker='x', c='k')












# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------







class Environment:
    '''
    Class containing details about the simulation environment, walls etc
    '''
    # Walls
    walls = []

    # Targets (Make child class)
    target_position = np.array([10.5,5])
    target_dist_thresh = 0.5**2

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
        for wall in Environment.walls:
            wall.instance_plot(ax)
        ax.scatter(Environment.target_position[0],Environment.target_position[1],s=20, c='g', marker='x')

    @staticmethod
    def draw_backdrop(ax):
        '''
        Called by Particle.animate_timestep to set background for each frame, 
         before drawing its particle objects over the top.
        An ax is passed in and we call different functions to draw environment elements
        '''
        # Hide border and ticks if using evac 
        if Environment.background_type == 'room':
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

        Environment.draw_background_colour(ax)
        Environment.draw_objects(ax)


class Wall(Environment):
    '''
    Encodes instance of a wall
    '''
    def __init__(self, a_position, b_position) -> None:
        super().__init__()
        self.a_position = a_position
        self.b_position = b_position
        self.wall_vec = b_position - a_position
        self.wall_length = np.sqrt(np.sum((self.wall_vec)**2))
        Environment.walls += [self]

    def __str__(self) -> str:
        return f"Wall_[{self.a_position}]_[{self.b_position}]."

    def instance_plot(self, ax):
        x_vals = np.array([self.a_position[0], self.b_position[0]])
        y_vals = np.array([self.a_position[1], self.b_position[1]])
        ax.plot(x_vals, y_vals, c='k')
        ax.scatter(x_vals,y_vals,s=20,c='r')

    def dist_to_wall(self, particle: Particle):
        '''
        Function taking a wall and particle with position.
        Returns the particle's closest distance to the wall, and the vector
        pointing from wall to particle (direction of repulsion force).
        '''
        x = particle.position
        a = self.a_position
        b = self.b_position
        vec = self.wall_vec # b-a
        length = self.wall_length
        '''
        # Check if not directly facing the wall by subtended angles
        # Nearest pole A
        ax = np.sqrt(np.sum((a-x)**2))
        dot_product_a = np.dot((a-x),(vec))/(length*ax)
        dot_product_a = np.clip(dot_product_a, -1.0, 1.0)
        tol = 1e-6
        xab = np.arccos(dot_product_a)
        if xab < np.pi/2:
            return ax, (a-x)
        # Nearest pole B
        bx = np.sqrt(np.sum((b-x)**2))
        dot_product_b = np.dot((b-x),(vec))/(length*bx)
        dot_product_b = np.clip(dot_product_b, -1.0, 1.0)
        xba = np.arccos(dot_product_b)
        if xba > np.pi/2:
            return bx, (b-x)
        '''
        # Check distance to point A (pole A)
        tolerance = 1e-2
        ax = np.linalg.norm(a - x)
        if ax < tolerance:
            # Particle is effectively at pole A
            return ax, (a - x)
        
        # Check distance to point B (pole B)
        bx = np.linalg.norm(b - x)
        if bx < tolerance:
            # Particle is effectively at pole B
            return bx, (b - x)
        
        # Projection of vector from A to particle onto the wall vector
        t = np.dot((x - a), vec) / (length * length)

        # If t < 0, the particle is closer to pole A
        if t < 0:
            return ax, -(a - x)
        # If t > 1, the particle is closer to pole B
        if t > 1:
            return bx, -(b - x)
        
        # Else 0 <= t <= 1, and the particle is perpendicular to the wall
        x_to_wall = (a-x) + t*vec
        return np.sqrt(np.sum(x_to_wall**2)), -x_to_wall
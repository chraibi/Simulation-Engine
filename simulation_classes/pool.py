import copy
import numpy as np
from .parents import Particle, Environment, Wall, Target

class Pool(Particle):
    '''
    Solid particle for the lattice of springs simulation
    '''
    # -------------------------------------------------------------------------
    # Attributes

    # Constants
    damping_constant = 2.5
    d = 0.051 # 5.1 cm diameter of pool ball
    k_ball = 30 #
    k_wall = 20
    max_collision_force = 50 # limit collision force by 50N upper limit

    # Counts
    num_red_potted = 0
    num_yellow_potted = 0
    white_potted = 0
    black_potted = 0

    # Initialisation
    def __init__(self, colour: str, position: np.ndarray = None, velocity: np.ndarray = None,
                id=None) -> None:
        '''
        Initialises a solid object, inheriting from the Particle class.
        '''
        super().__init__(position, velocity,id)

        self.mass = 0.14
        self.colour = colour

        # Ensure prototype for child class exists, callable by its name as a string only
        prototype = copy.copy(self)
        Particle.prototypes[self.__class__.__name__] = prototype

    def create_instance(self,id):
        ''' Used to create instance of the same class as self, without referencing class. '''
        return Pool(id=id)
    
    def unalive(self):
        ''' 
        Sets the class instance with this id to be not alive, decrements the class count.
        '''
        self.alive=0
        Particle.pop_counts_dict[self.__class__.__name__] -= 1
        Particle.kill_count += 1
        Particle.kill_record[Particle.current_step] = Particle.kill_count
        # Update colour records
        if self.colour == 'r':
            Pool.num_red_potted += 1
        elif self.colour == 'y':
            Pool.num_yellow_potted += 1
        elif self.colour == 'w':
            Pool.white_potted = 1
        elif self.colour == 'k':
            Pool.black_potted = 1

    # -------------------------------------------------------------------------
    # Main force model
    
    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''
        # Instantiate force term
        force_term = np.zeros(2)

        # Go through targets and check distance to escape threshold
        # If escape possible, unalive self, and cease update function
        if Environment.targets is not []:
            for target in Environment.targets:
                dist, _ = target.dist_to_target(self)
                if dist < self.d:
                    self.unalive()
                    return 1

        # Repulsion from balls - in range scales with 1/d up to limit
        for ball in Pool.iterate_class_instances():
            if ball == self:
                continue
            elif self.dist(ball) < self.d: # 1 diameter between ball centres -> collision
                repulsion = np.min( [self.k_ball/(np.sqrt(self.dist(ball))), self.max_collision_force] )
                force_term += - self.unit_dirn(ball)*repulsion
            else:
                continue

        # Reflection from walls - in range, walls act like stiff springs
        for wall in Environment.walls:
            dist, _ = wall.dist_to_wall(self)
            if dist < self.d/2: # 1 radius to wall -> collision F = ke
                compression = self.d/2 - dist
                force_term += wall.perp_vec * self.k_wall * compression

        # Damping force to simulate friction
        force_term += -self.velocity * Pool.damping_constant

        # Update acceleration = Force / mass
        self.acceleration = force_term / self.mass

        return 0
    


    # -------------------------------------------------------------------------
    # CSV utilities

    def write_csv_list(self):
        '''
        Format for compressing each Star instance into CSV.
        '''
        # Individual child instance info
        return [self.id, self.colour, \
                self.position[0], self.position[1], \
                self.last_position[0],self.last_position[1],
                self.velocity[0], self.velocity[1],
                self.acceleration[0], self.acceleration[1] ]

    def read_csv_list(self, system_state_list, idx_shift):
        '''
        Format for parsing the compressed Star instances from CSV.
        '''
        self.colour = str(system_state_list[idx_shift+1])
        self.position = np.array([float(system_state_list[idx_shift+2]), \
                                    float(system_state_list[idx_shift+3])])
        self.last_position = np.array([float(system_state_list[idx_shift+4]), \
                                    float(system_state_list[idx_shift+5])])
        self.velocity = np.array([float(system_state_list[idx_shift+6]), \
                                    float(system_state_list[idx_shift+7])])
        self.acceleration = np.array([float(system_state_list[idx_shift+8]), \
                                    float(system_state_list[idx_shift+9])])
        # Update idx shift to next id and return
        return idx_shift+10
    
    # -------------------------------------------------------------------------
    # Animation utilities

    def instance_plot(self, ax, com=None, scale=None):
        ''' 
        Plots individual Pool ball particle onto existing axis, and plots score
        '''
        size = 15**2
        ax.scatter(self.position[0], self.position[1], color=self.colour, s=size)

        # Plot scores if not already done
        if ax.texts:
            pass
        else:
            fontsize = 15**2
            if Pool.num_red_potted > 0:
                ax.text(x=0.2,y=-Pool.d, s=f"{Pool.num_red_potted}", fontsize=fontsize)
            if Pool.num_yellow_potted > 0:
                ax.text(x=0.4,y=-Pool.d, s=f"{Pool.num_yellow_potted}", fontsize=fontsize)
            if Pool.white_potted:
                ax.text(x=0.6,y=-Pool.d, s=f"{Pool.white_potted}", fontsize=fontsize)
            if Pool.black_potted:
                ax.text(x=0.8,y=-Pool.d, s=f"{Pool.black_potted}", fontsize=fontsize)







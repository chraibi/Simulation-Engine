import numpy as np

from parents import Particle, Environment, Wall

class Human(Particle):
    '''
    Human particle for crowd simulation.
    '''
    # -------------------------------------------------------------------------
    # Attributes

    personal_space = 1 # metres - 2 rulers between centres
    personal_space_repulsion = 100 # Newtons

    wall_dist_thresh = 0.5
    wall_repulsion = 100

    target_attraction = 1000

    random_force = 30
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None) -> None:
        '''
        Initialises a Human, inheriting from the Particle class.
        '''
        super().__init__(position, velocity)

        # Prey specific attributes
        self.mass = 60
        self.max_speed = 1.5

        # Find closest exit target
        # TODO: assign each human a target on initialisation, using shortest distance

    # -------------------------------------------------------------------------
    # Distance utilities

    # -------------------------------------------------------------------------
    # Main force model 

    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''
        if Environment.target_position is not None:
            dist = np.sum((Environment.target_position - self.position)**2)
            if dist < Environment.target_dist_thresh:
                self.unalive()
                return 1
        
        # Instantiate force term
        force_term = np.zeros(2)

        # Human repulsion force - currently scales with 1/d^2
        for human in Human.iterate_class_instances():
            if human == self:
                continue
            elif self.dist(human) < self.personal_space:
                force_term += - self.unit_dirn(human)*(self.personal_space_repulsion/(np.sqrt(self.dist(human))))

        # Attraction to target
        if Environment.target_position is not None:
            vec = Environment.target_position - self.position
            dirn = (vec)/np.linalg.norm(vec)
            force_term += dirn * self.target_attraction

        # Repulsion from walls - scales with 1/d^2
        for wall in Environment.walls:
            dist, dirn = wall.dist_to_wall(self)
            if dist < self.wall_dist_thresh:
                force_term += dirn * (self.wall_repulsion/(dist**3))

        # Random force - stochastic noise
        # Generate between [0,1], map to [0,2] then shift to [-1,1]
        force_term += ((np.random.rand(2)*2)-1)*self.random_force

        # Update acceleration = Force / mass
        self.acceleration = force_term / self.mass

        return 0
    
    # -------------------------------------------------------------------------
    # CSV utilities

    def write_csv_list(self):
        '''
        Format for compressing each Human instance into CSV.
        '''
        # Individual child instance info
        return [self.id, \
                self.position[0], self.position[1], \
                self.last_position[0],self.last_position[1],
                self.velocity[0], self.velocity[1],
                self.acceleration[0], self.acceleration[1] ]

    def read_csv_list(self, system_state_list, idx_shift):
        '''
        Format for parsing the compressed Human instances from CSV.
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

    def instance_plot(self, ax, com=None, scale=None):
        ''' 
        Plots individual Prey particle onto existing axis. 
        '''
        # Get plot position in frame
        plot_position = self.position
        if (com is not None) and (scale is not None):
            plot_position = self.orient_to_com(com, scale)
        
        ax.scatter(plot_position[0],plot_position[1],s=15**2,c='b')

        



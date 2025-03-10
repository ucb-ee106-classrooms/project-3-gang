import numpy as np
import time

class Landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.pos = np.array([x, y, z])

class Environment:
    def __init__(self, dynamics, controller, landmark, observer = None, is_noise = False):
        """
        Initializes a simulation environment
        Args:
            dynamics (Dynamics): system dynamics object
            controller (Controller): system controller object
            observer (Observer): system state estimation object
        """
        #store system parameters
        self.dynamics = dynamics
        self.controller = controller
        self.observer = observer
        self.landmark = landmark
        
        #define environment parameters
        self.iter = 0 #number of iterations
        self.t = 0 #time in seconds 
        self.clock_zero = time.time()
        self.done = False
        
        #Store system state
        self.x = self.dynamics.get_state() #Actual state of the system
        self.x0 = self.x #store initial condition for use in reset
        self.xObsv = None #state as read by the observer
        
        #Define simulation parameters
        self.SIM_FREQ = 10000 #integration frequency in Hz
        self.CONTROL_FREQ = 500 #control frequency in Hz
        self.SIMS_PER_STEP = self.SIM_FREQ//self.CONTROL_FREQ
        self.TOTAL_SIM_TIME = 6 #total simulation time in s
        
        #Define history arrays
        self.xHist = np.zeros((self.dynamics.stateDimn, self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1))
        self.uHist = np.zeros((self.dynamics.inputDimn, self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1))
        self.tHist = np.zeros((1, self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1))
        self.obsHist = np.zeros((3, self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1))
        
        # Determine whether or not we want to have noise in our system
        self.is_noise = is_noise

        # NOTE: If you want to generate data with process and or measurement noise change these values!
        self.w = 0
        self.v = 0
        
    def reset(self):
        """
        Reset the gym environment to its inital state.
        """
        #Reset gym environment parameters
        self.iter = 0 #number of iterations
        self.t = 0 #time in seconds
        self.done = False
        
        #Reset system state
        self.x = self.x0 #retrieves initial condiiton
        self.xObsv = None #reset observer state
        
        #Define history arrays
        self.xHist = np.zeros((self.dynamics.stateDimn, self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1))
        self.uHist = np.zeros((self.dynamics.inputDimn, self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1))
        self.tHist = np.zeros((1, self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1))
        self.obsHist = np.zeros((2, self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1))

    def step(self):
        """
        Step the sim environment by one integration
        """
        #retrieve current state information
        # self._get_observation() #updates the observer
        
        #solve for the control input using the observed state
        self.controller.eval_input(self.t)
        
        #Zero order hold over the controller frequency
        for i in range(self.SIMS_PER_STEP):
            self.dynamics.integrate(self.controller.get_input(), self.t, 1/self.SIM_FREQ) #integrate dynamics
            self.t += 1/self.SIM_FREQ #increment the time

            # generates the process noise
            if self.is_noise:
                xw = np.random.normal(0, self.w, size=self.x.shape)
                self.x = xw + self.x.astype(np.float64)

            x_t = self.x.reshape((self.dynamics.stateDimn, ))
            drone_position = x_t[:3]
            
            self.y = np.array([np.linalg.norm(self.landmark.pos - drone_position.reshape((3,))), self.x[3][0]]).reshape((2, ))
            # generates the measurement noise
            if self.is_noise:
                yv = np.random.normal(0, self.v, size=self.y.shape)
                self.y = yv + self.y.astype(np.float64)

            u_t = (self.controller.get_input()).reshape((self.dynamics.inputDimn, ))
            
        #update the deterministic system data, iterations, and history array
        self._update_data()        
    
    def _update_data(self):
        """
        Update history arrays and deterministic state data
        """
        #append the input, time, and state to their history queues
        self.xHist[:, self.iter] = self.x.reshape((self.dynamics.stateDimn, ))
        self.uHist[:, self.iter] = (self.controller.get_input()).reshape((self.dynamics.inputDimn, ))
        self.tHist[:, self.iter] = self.t
        # print(drone_position.reshape((3,))) 
        self.obsHist[:, self.iter] = self.y
        
        #update the actual state of the system
        self.x = self.dynamics.get_state()
        
        #update the number of iterations of the step function
        self.iter +=1
    
    def _get_observation(self):
        """
        Updates self.xObsv using the observer data
        Useful for debugging state information.
        """
        self.xObsv = self.observer.get_state()
        # print("current orientation: ", self.observer.get_orient())
    
    def _get_reward(self):
        """
        Calculate the total reward for ths system and update the reward parameter.
        Only implement for use in reinforcement learning.
        """
        return 0
    
    def _is_done(self):
        """
        Check if the simulation is complete
        Returns:
            boolean: whether or not the time has exceeded the total simulation time
        """
        #check current time with respect to simulation time
        if self.t >= self.TOTAL_SIM_TIME:
            return True
        return False
    
    def run(self):
        self.reset()
        while not self._is_done():
            # print("Simulation Time Remaining: ", self.TOTAL_SIM_TIME - self.t)
            self.step() #step the environment while not done

        return self.xHist, self.uHist, self.tHist, self.obsHist
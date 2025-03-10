import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

class Dynamics:
    """
    Skeleton class for system dynamics
    Includes methods for returning state derivatives, plots, and animations
    """
    def __init__(self, x0, stateDimn, inputDimn, relDegree = 1):
        """
        Initialize a dynamics object
        Args:
            x0 (stateDimn x 1 numpy array): initial condition state vector
            stateDimn (int): dimension of state vector
            inputDimn (int): dimension of input vector
            relDegree (int, optional): relative degree of system. Defaults to 1.
        """
        self.stateDimn = stateDimn
        self.inputDimn = inputDimn
        self.relDegree = relDegree
        
        #Store the state and input
        self._x = x0
        self._u = None
    
    def get_state(self):
        """
        Retrieve the state vector
        """
        return self._x
        
    def deriv(self, x, u, t):
        """
        Returns the derivative of the state vector
        Args:
            x (stateDimn x 1 numpy array): current state vector at time t
            u (inputDimn x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
        Returns:
            xDot: state_dimn x 1 derivative of the state vector
        """
        return np.zeros((self.state_dimn, 1))
    
    def integrate(self, u, t, dt):
        """
        Integrates system dynamics using Euler integration
        Args:
            u (inputDimn x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
            dt (float): time step for integration
        Returns:
            x (stateDimn x 1 numpy array): state vector after integrating
        """
        #integrate starting at x
        self._x = self.get_state() + self.deriv(self.get_state(), u, t)*dt
        return self._x
    
    def get_plots(self, x, u, t):
        """
        Function to show plots specific to this dynamic system.
        Args:
            x ((stateDimn x N) numpy array): history of N states to plot
            u ((inputDimn x N) numpy array): history of N inputs to plot
            t ((1 x N) numpy array): history of N times associated with x and u
        """
        pass
    
    def show_animation(self, x, u, t):
        """
        Function to play animations specific to this dynamic system.
        Args:
            x ((stateDimn x N) numpy array): history of N states to plot
            u ((inputDimn x N) numpy array): history of N inputs to plot
            t ((1 x N) numpy array): history of N times associated with x and u
        """
        pass

class QuadDyn(Dynamics):
    def __init__(self, x0 = np.zeros((8, 1)), stateDimn = 8, inputDimn = 2, relDegree = 2, m = 0.92, Ixx = 0.0023, l = 0.15):
        """
        Init function for a Planar quadrotor system.
        State Vector: X = [x, y, z, theta, x_dot, y_dot, z_dot, theta_dot]
        Input Vector: U = [F, M]
        
        Args:
            x0 ((8 x 1) NumPy Array): initial state (x, y, z, theta, x_dot, y_dot, z_dot, theta_dot)
            stateDimn (int): dimension of state vector
            inputDimn (int): dimension of input vector
            relDegree (int, optional): relative degree of system
            m (float): mass of quadrotor in kg
            Ixx (float): moment of inertia about x axis of quadrotor
            l (float): length of one arm of quadrotor
        """
        super().__init__(x0, stateDimn, inputDimn, relDegree)
        
        #store physical parameters
        self._m = m
        self._Ixx = Ixx
        self._g = 9.81 
        self._l = l

        #store an animation reference for Colab
        self.anim = None
    
    def deriv(self, X, U, t):
        """
        Returns the derivative of the state vector
        Args:
            X (8 x 1 numpy array): current state vector at time t
            U (2 x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
        Returns:
            xDot: state_dimn x 1 derivative of the state vector
        """
        #unpack the input vector
        F, M = U[0, 0], U[1, 0]
        F = max(0, F) #CUT OFF THE FORCE AT ZERO!
        
        #unpack the state vector
        x_dot, y_dot, z_dot = X[4, 0], X[5, 0], X[6, 0] #velocities
        theta, theta_dot = X[3, 0], X[7, 0] #orientations
        
        #calculates the second time derivatives of each
        x_ddot = (-np.sin(theta)*F)/self._m
        y_ddot = 0
        
        z_ddot = (np.cos(theta)*F - self._m*self._g)/self._m
        theta_ddot = M/self._Ixx
        
        #construct and returns the state vector        
        return np.array([[x_dot, y_dot, z_dot, theta_dot, x_ddot, y_ddot, z_ddot, theta_ddot]]).T
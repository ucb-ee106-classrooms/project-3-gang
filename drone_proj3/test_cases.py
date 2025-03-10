from dynamics import QuadDyn
from controller import PlanarQrotorOrchestrated
from trajectory import InputTrajectory
from environment import Environment, Landmark

import numpy as np

def test_up_and_down():
    SIM_LEN = 6 # simulation time (seconds)

    landmark = Landmark(0, 5, 5)

    #system initial condition
    x0 = np.array([[10, 0, 1, 0, 0, 0, 0, 0]]).T

    dynamics = QuadDyn(x0)

    def up_and_down_traj(t):
        # defines the input trajectory to fly the drone up and then back down.
        return (
            0.1 * (1 - (2 / SIM_LEN) * t) + dynamics._m * dynamics._g, 
            0)

    inp_traj = InputTrajectory(up_and_down_traj) # input-space trajectory

    #create a planar quadrotor controller
    controller = PlanarQrotorOrchestrated(trajectory = inp_traj)

    #create a simulation environment
    env = Environment(dynamics, controller, landmark)
    env.reset()

    #run the simulation
    return env.run()

def test_loop():
    landmark = Landmark(0, 5, 5)

    #system initial condition
    x0 = np.array([[10, 0, 1, 0, 0, 0, 0, 0]]).T

    dynamics = QuadDyn(x0)

    def u2(t):
        # define the moment signal
        period = 3
        return -((2 * np.pi / period) ** 2) * np.sin((2 * np.pi / period) * t - np.pi/2) * (np.pi / 4) / 900 

    def inp_traj(t):
        # define the input trajectory to fly the drone in a loop-like flight path
        if t < 3:
            return (1 + dynamics._m * dynamics._g, u2(t))
        else:
            return (5 * (t - 3) + dynamics._m * dynamics._g, -u2(t))

    inp_traj = InputTrajectory(inp_traj) # input-space trajectory

    controller = PlanarQrotorOrchestrated(trajectory = inp_traj)

    #create a simulation environment
    env = Environment(dynamics, controller, landmark)
    env.reset()

    return env.run()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # xHist, uHist, tHist, obsHist = test_up_and_down()
    xHist, uHist, tHist, obsHist = test_loop()
    timesteps = list(range(len(xHist[0, :])))

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, xHist[0, :], label='x')
    plt.plot(timesteps, xHist[1, :], label='y')
    plt.plot(timesteps, xHist[2, :], label='z')
    plt.legend()
    plt.title('Drone 3d position')
    plt.xlabel('Simulation timestep')
    plt.ylabel("Coordinate values (m)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, uHist[0, :], label='F')
    plt.legend()
    plt.title('Force input')
    plt.xlabel('Simulation timestep')
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, uHist[1, :], label='M')
    plt.legend()
    plt.title('Moment input')
    plt.xlabel('Simulation timestep')
    plt.ylabel("Moment (N-m)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, obsHist[0, :], label='distance')
    plt.legend()
    plt.title('Observation model')
    plt.xlabel('Simulation timestep')
    plt.ylabel("Magnitude (m)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, xHist[3, :], label='phi', color='red')
    plt.plot(timesteps, obsHist[1, :], label="observed phi")
    plt.legend()
    plt.title('Drone orientation')
    plt.xlabel('Simulation timestep')
    plt.ylabel("Magnitude (rad)")
    plt.grid(True)
    plt.show()
######################################################################################################
# Project 3
# Authors: Nima Rahmanian, Karim El-Refai
# Note: this code is adapted from EECS 106B Homework 2 Problem Implementing Control Lyapunov Functions
######################################################################################################

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from pyplot3d.uav import Uav
from pyplot3d.utils import ypr_to_R

from test_cases import test_up_and_down, test_loop

def update_plot(drone_trajectory):
    def helper(i):
        uav_plot.draw_at(drone_trajectory[0][:, i], drone_trajectory[1][:, :, i])

        circle = Circle((0, 0), radius=3, color='purple')
        ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=20, zdir="y")

        ax.set(xlim3d=(-30, 30), xlabel='X')
        ax.set(ylim3d=(-30, 0), ylabel='Y')
        ax.set(zlim3d=(0, 30), zlabel='Z')

        return ln,

    return helper

# Set up plotting
plt.style.use('seaborn')

fig = plt.figure()
plt.plot()
ax = fig.add_subplot(projection="3d")
ax.set(xlim3d=(-30, 30), xlabel='X')
ax.set(ylim3d=(-30, 0), ylabel='Y')
ax.set(zlim3d=(0, 30), zlabel='Z')

uav_plot = Uav(ax, arm_length = 5, scaling_factor = 20)

x = []
y = []
z = []
ln, = ax.plot(x, y, z, '-')

def main():

    # TODO: choose the motion to simulate
    # xHist, _, _ = test_up_and_down()
    xHist, uHist, tHist, obsHist = test_loop()

    x = xHist[:3, ::30]
    phi = xHist[3, ::30]
    SIM_LEN = x.shape[1]
    R = np.zeros((3, 3, SIM_LEN))
    for i in range(SIM_LEN):
        ypr = np.array([0, -phi[i], 0])
        R[:, :, i] = ypr_to_R(ypr, degrees=False)
    drone_trajectory = (x, R)
    
    animation = FuncAnimation(fig, update_plot(drone_trajectory), frames=SIM_LEN, interval=10)
    plt.show()

    # want to remove y and y_dot from the state vector
    xHist = np.delete(xHist, (1,5),0)
    dataHist = np.vstack((tHist, xHist))
    dataHist = np.vstack((dataHist, uHist))
    dataHist = np.vstack((dataHist,obsHist))
    # this is a (N,12) where it's time, x, u, then obs 
    dataHist = dataHist.T

    with open('data.npy', 'wb') as f:
        np.save(f, dataHist)

if __name__ == '__main__':
    main()
import rospy
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['font.family'] = ['FreeSans', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 14


class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        d : float
            Half of the track width (m) of TurtleBot3 Burger.
        r : float
            Wheel radius (m) of the TurtleBot3 Burger.
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][0] is timestamp (s),
            u[i][1] is left wheel rotational speed (rad/s), and
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is timestamp (s),
            x[i][1] is bearing (rad),
            x[i][2] is translational position in x (m),
            x[i][3] is translational position in y (m),
            x[i][4] is left wheel rotational position (rad), and
            x[i][5] is right wheel rotational position (rad).
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][0] is timestamp (s),
            y[i][1] is translational position in x (m) when freeze_bearing:=true,
            y[i][1] is distance to the landmark (m) when freeze_bearing:=false,
            y[i][2] is translational position in y (m) when freeze_bearing:=true, and
            y[i][2] is relative bearing (rad) w.r.t. the landmark when
            freeze_bearing:=false.
        x_hat : list
            A list of estimated system states. It should follow the same format
            as x.
        dt : float
            Update frequency of the estimator.
        fig : Figure
            matplotlib Figure for real-time plotting.
        axd : dict
            A dictionary of matplotlib Axis for real-time plotting.
        ln* : Line
            matplotlib Line object for ground truth states.
        ln_*_hat : Line
            matplotlib Line object for estimated states.
        canvas_title : str
            Title of the real-time plot, which is chosen to be estimator type.
        sub_u : rospy.Subscriber
            ROS subscriber for system inputs.
        sub_x : rospy.Subscriber
            ROS subscriber for system states.
        sub_y : rospy.Subscriber
            ROS subscriber for system outputs.
        tmr_update : rospy.Timer
            ROS Timer for periodically invoking the estimator's update method.

    Notes
    ----------
        The frozen bearing is pi/4 and the landmark is positioned at (0.5, 0.5).
    """
    # noinspection PyTypeChecker
    def __init__(self):
        self.d = 0.08
        self.r = 0.033
        self.u = []
        self.x = []
        self.y = []
        self.x_hat = []  # Your estimates go here!
        self.dt = 0.1
        self.fig, self.axd = plt.subplot_mosaic(
            [['xy', 'phi'],
             ['xy', 'x'],
             ['xy', 'y'],
             ['xy', 'thl'],
             ['xy', 'thr']], figsize=(20.0, 10.0))
        self.ln_xy, = self.axd['xy'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_xy_hat, = self.axd['xy'].plot([], 'o-c', label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_y, = self.axd['y'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_y_hat, = self.axd['y'].plot([], 'o-c', label='Estimated')
        self.ln_thl, = self.axd['thl'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thl_hat, = self.axd['thl'].plot([], 'o-c', label='Estimated')
        self.ln_thr, = self.axd['thr'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thr_hat, = self.axd['thr'].plot([], 'o-c', label='Estimated')
        self.canvas_title = 'N/A'
        self.sub_u = rospy.Subscriber('u', Float32MultiArray, self.callback_u)
        self.sub_x = rospy.Subscriber('x', Float32MultiArray, self.callback_x)
        self.sub_y = rospy.Subscriber('y', Float32MultiArray, self.callback_y)
        self.tmr_update = rospy.Timer(rospy.Duration(self.dt), self.update)

        self.execution_time = 0.0
        
    def get_execution_time(self):
        return self.execution_time

    def callback_u(self, msg):
        self.u.append(msg.data)

    def callback_x(self, msg):
        self.x.append(msg.data)
        if len(self.x_hat) == 0:
            self.x_hat.append(msg.data)

    def callback_y(self, msg):
        self.y.append(msg.data)

    def update(self, _):
        raise NotImplementedError

    def plot_init(self):
        self.axd['xy'].set_title(self.canvas_title)
        self.axd['xy'].set_xlabel('x (m)')
        self.axd['xy'].set_ylabel('y (m)')
        self.axd['xy'].set_aspect('equal', adjustable='box')
        self.axd['xy'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].legend()
        self.axd['y'].set_ylabel('y (m)')
        self.axd['y'].legend()
        self.axd['thl'].set_ylabel('theta L (rad)')
        self.axd['thl'].legend()
        self.axd['thr'].set_ylabel('theta R (rad)')
        self.axd['thr'].set_xlabel('Time (s)')
        self.axd['thr'].legend()
        plt.tight_layout()

    def plot_update(self, _):
        self.plot_xyline(self.ln_xy, self.x)
        self.plot_xyline(self.ln_xy_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_yline(self.ln_y, self.x)
        self.plot_yline(self.ln_y_hat, self.x_hat)
        self.plot_thlline(self.ln_thl, self.x)
        self.plot_thlline(self.ln_thl_hat, self.x_hat)
        self.plot_thrline(self.ln_thr, self.x)
        self.plot_thrline(self.ln_thr_hat, self.x_hat)

    def plot_xyline(self, ln, data):
        if len(data):
            x = [d[2] for d in data]
            y = [d[3] for d in data]
            ln.set_data(x, y)
            self.resize_lim(self.axd['xy'], x, y)

    def plot_philine(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            phi = [d[1] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            x = [d[2] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_yline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            y = [d[3] for d in data]
            ln.set_data(t, y)
            self.resize_lim(self.axd['y'], t, y)

    def plot_thlline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thl = [d[4] for d in data]
            ln.set_data(t, thl)
            self.resize_lim(self.axd['thl'], t, thl)

    def plot_thrline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thr = [d[5] for d in data]
            ln.set_data(t, thr)
            self.resize_lim(self.axd['thr'], t, thr)

    # noinspection PyMethodMayBeStatic
    def resize_lim(self, ax, x, y):
        xlim = ax.get_xlim()
        ax.set_xlim([min(min(x) * 1.05, xlim[0]), max(max(x) * 1.05, xlim[1])])
        ylim = ax.get_ylim()
        ax.set_ylim([min(min(y) * 1.05, ylim[0]), max(max(y) * 1.05, ylim[1])])

    def print_accuracy(self):
        valid_timesteps = len(self.x)

        xs = np.array(self.x)
        # xs_timesteps = xs[:, 0]
        xs = xs[:, 1:]
        x_hats = np.array(self.x_hat)
        # x_hats_timesteps = x_hats[:, 0]
        x_hats = x_hats[:valid_timesteps, 1:]
        # print(xs_timesteps, x_hats_timesteps, flush=True)

       
        RMSE = np.sqrt(np.mean(np.power(xs - x_hats, 2), axis = 0))

        print(f"RMSE: {RMSE}")


class OracleObserver(Estimator):
    """Oracle observer which has access to the true state.

    This class is intended as a bare minimum example for you to understand how
    to work with the code.

    Example
    ----------
    To run the oracle observer:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=oracle_observer \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Oracle Observer'

    def update(self, _):
        self.x_hat.append(self.x[-1])


class DeadReckoning(Estimator):
    """Dead reckoning estimator.

    Your task is to implement the update method of this class using only the
    u attribute and x0. You will need to build a model of the unicycle model
    with the parameters provided to you in the lab doc. After building the
    model, use the provided inputs to estimate system state over time.

    The method should closely predict the state evolution if the system is
    free of noise. You may use this knowledge to verify your implementation.

    Example
    ----------
    To run dead reckoning:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=dead_reckoning \
            noise_injection:=true \
            freeze_bearing:=false
    For debugging, you can simulate a noise-free unicycle model by setting
    noise_injection:=false.
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Dead Reckoning'

    def g(self, x, u):
        return x + (np.array([
            [-self.r / (2 * self.d), self.r / (2 * self.d)],
            [self.r / 2 * np.cos(x[0]), self.r / 2 * np.cos(x[0])],
            [self.r / 2 * np.sin(x[0]), self.r / 2 * np.sin(x[0])],
            [1, 0],
            [0, 1]
        ]) @ u) * self.dt

    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may ONLY use self.u and self.x[0] for estimation

            start_time = rospy.Time.now()

            x_hat = np.array(self.x[0][1:])
            for k in range(len(self.u)):
                u_k = np.array(self.u[k][1:])
                x_hat = self.g(x_hat, u_k)
            #     x_prev = x_hat
                
            #     phi = x_prev[1]  
            #     u_L = u_k[1] 
            #     u_R = u_k[2] 
                
            #     phi_dot = (self.r/(2*self.d))*(u_R - u_L)
            #     x_dot = (self.r/2)*(u_L + u_R)*np.cos(phi)
            #     y_dot = (self.r/2)*(u_L + u_R)*np.sin(phi)
            #     theta_L_dot = u_L
            #     theta_R_dot = u_R
                
            #     dt = u_k[0] - x_prev[0]  
            #     new_state = [
            #         u_k[0],            
            #         x_prev[1] + phi_dot*dt,      
            #         x_prev[2] + x_dot*dt,         
            #         x_prev[3] + y_dot*dt,           
            #         x_prev[4] + theta_L_dot*dt,    
            #         x_prev[5] + theta_R_dot*dt      
            #     ]
            # print(x_hat)
            self.x_hat.append([self.x[-1][0]] + list(x_hat))

            end_time = rospy.Time.now()
            self.execution_time += (end_time - start_time).to_sec()

class KalmanFilter(Estimator):
    """Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    linear unicycle model at the default bearing of pi/4. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive Kalman filter update rule.

    Attributes:
    ----------
        phid : float
            Default bearing of the turtlebot fixed at pi / 4.

    Example
    ----------
    To run the Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=kalman_filter \
            noise_injection:=true \
            freeze_bearing:=true
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Kalman Filter'
        self.phid = np.pi / 4
        # TODO: Your implementation goes here!
        # You may define the A, C, Q, R, and P matrices below.
        self.A = np.eye(4)
        self.B = np.array([
            [self.r/2 * np.cos(self.phid), self.r/2 * np.cos(self.phid)], 
            [self.r/2 * np.sin(self.phid), self.r/2 * np.sin(self.phid)],  
            [1, 0],  
            [0, 1]   
        ]) * self.dt
        self.C = np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0]   
        ])
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 0.1  
        self.P = np.eye(4) * 0.1 

    # noinspection DuplicatedCode
    # noinspection PyPep8Naming
    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may use self.u, self.y, and self.x[0] for estimation
            start_time = rospy.Time.now()

            x_hat = np.array(self.x[0])
            for k in range(len(self.u)):
                u_k = np.array(self.u[k])  
                y_k = np.array(self.y[k])  
                x_prev = x_hat[[2, 3, 4, 5]] 
                dt = u_k[0] - x_prev[0] 
                x_pred = self.A @ x_prev + self.B @ u_k[1:]
                P_pred = self.A @ self.P @ self.A.T + self.Q
                S = self.C @ P_pred @ self.C.T + self.R  
                K = P_pred @ self.C.T @ np.linalg.inv(S)  
                y_pred = self.C @ x_pred  
                x_update = x_pred + K @ (y_k[1:] - y_pred)  
                self.P = (np.eye(4) - K @ self.C) @ P_pred
                x_hat = np.array([u_k[0], self.phid, x_update[0], x_update[1], x_update[2], x_update[3]])
            self.x_hat.append(list(x_hat))

            end_time = rospy.Time.now()
            self.execution_time += (end_time - start_time).to_sec()


# noinspection PyPep8Naming
class ExtendedKalmanFilter(Estimator):
    """Extended Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    unicycle model and linearize it at every operating point. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive extended Kalman filter update rule.

    Hint: You may want to reuse your code from DeadReckoning class and
    KalmanFilter class.

    Attributes:
    ----------
        landmark : tuple
            A tuple of the coordinates of the landmark.
            landmark[0] is the x coordinate.
            landmark[1] is the y coordinate.

    Example
    ----------
    To run the extended Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=extended_kalman_filter \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Extended Kalman Filter'
        self.landmark = (0.5, 0.5)
        # TODO: Your implementation goes here!
        # You may define the Q, R, and P matrices below.
        self.Q = 0.05 * np.eye(5)
        self.R = 0.01 * np.eye(2)
        self.P = 0.05 * np.eye(5)

    
    def g(self, x, u):
        return x + (np.array([
            [-self.r / (2 * self.d), self.r / (2 * self.d)],
            [self.r / 2 * np.cos(x[0]), self.r / 2 * np.cos(x[0])],
            [self.r / 2 * np.sin(x[0]), self.r / 2 * np.sin(x[0])],
            [1, 0],
            [0, 1]
        ]) @ u) * self.dt

    def h(self, x):
        return np.array([self.dist_to_landmark(x), x[0] - np.arctan2((self.landmark[0] - x[1]), (self.landmark[1] - x[2]))])
        # return np.array([self.dist_to_landmark(x), x[0]])

    def dist_to_landmark(self, x):
        return np.sqrt( (self.landmark[0] - x[1]) ** 2 + (self.landmark[1] - x[2]) ** 2)

    def approx_A(self, x, u):
        return np.eye(5) + np.array([
            [0, 0, 0, 0, 0],
            [-np.sin(x[0]) * self.r / 2 * (u[0] + u[1]) * self.dt, 0, 0, 0, 0],
            [np.cos(x[0]) * self.r / 2 * (u[0] + u[1]) * self.dt, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

    def approx_C(self, x):
        l_y_diff = self.landmark[1] - x[2]
        l_x_diff = self.landmark[0] - x[1]
        return np.array([
            [0, (x[1] - self.landmark[0]) / self.dist_to_landmark(x), (x[2] - self.landmark[1]) / self.dist_to_landmark(x), 0, 0],
            [1, -1 / (1 + (l_y_diff / l_x_diff) ** 2) * (l_y_diff) / (l_x_diff) ** 2, 1 / (1 + (l_y_diff / l_x_diff) ** 2) / (l_x_diff), 0, 0]
        ])

    # noinspection DuplicatedCode
    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # print(self.y)
            # TODO: Your implementation goes here!
            # You may use self.u, self.y, and self.x[0] for estimation
            # p sure for EKF we use most recent estimate so I'll use x_hat[-1]

            start_time = rospy.Time.now()
            
            x_t = np.array(self.x_hat[-1][1:])
            u_t = np.array(self.u[-1][1:])

            x_t_1 = self.g(x_t, u_t) #state extrapolation
            self.A = self.approx_A(x_t, u_t) #dynamics linearization
            self.P = self.A @ self.P @ self.A.T + self.Q #covariance extrapolation

            self.C = self.approx_C(x_t_1) #measurement linearization
            K = self.P @ self.C.T @ linalg.pinv(self.C @ self.P @ self.C.T + self.R) #Kalman gain
            x_t_1 = x_t_1 + K @ (np.array(self.y[-1][1:]) - self.h(x_t_1)) #state update
            self.P = (np.eye(5) - K @ self.C) @ self.P #covariance update

            self.x_hat.append([self.x[-1][0]] + list(x_t_1))
            
            end_time = rospy.Time.now()
            self.execution_time += (end_time - start_time).to_sec()


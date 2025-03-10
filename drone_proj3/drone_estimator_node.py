#!/usr/bin/env python3
# import rospy
from drone_estimator import \
    OracleObserver, DeadReckoning, ExtendedKalmanFilter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

plt.show(block=True)

parser = argparse.ArgumentParser()
parser.add_argument('--estimator', help='the estimator you want to use')

def spin(estimator):
    """
    Parameters
    ----------
    estimator : Estimator
        The instance of the estimator

    Returns
    -------
        None
    """

    # noinspection PyUnusedLocal
    estimator.run()
    anim = FuncAnimation(
        estimator.fig,
        estimator.plot_update,
        init_func=estimator.plot_init,
        cache_frame_data=False)
    plt.show(block=True)


def main():
    """Entry point of the estimator node.

    Returns
    -------
        None
    """
    args = parser.parse_args()
    estimator_type = args.estimator
    if estimator_type == 'oracle':
        estimator = OracleObserver(is_noisy=True)
    elif estimator_type == 'dr':
        estimator = DeadReckoning(is_noisy=True)
    elif estimator_type == 'kf':
        raise RuntimeError(
            f'Estimator type: {estimator_type} is not supported for the quadrotor!')
    elif estimator_type == 'ekf':
        estimator = ExtendedKalmanFilter(is_noisy=True)
    else:
        raise RuntimeError(
            'Estimator type {} not supported'.format(estimator_type))
    print('Invoking estimator {}...'.format(estimator_type))
    spin(estimator)


if __name__ == '__main__':
    main()

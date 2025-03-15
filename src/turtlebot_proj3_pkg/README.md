# proj3_pkg

Your task is to implement the `Estimator.py` class.
Specifically, you need to implement the `update` method for `DeadReckoning`
class, and `ExtendedKalmanFilter` class, respectively.

The starter code is only tested with `python3.8`.

To install the dependencies:
```angular2html
pip install -r -U requirements.txt
```

To run `DeadReckoning` estimator:
```angular2html
roslaunch proj3_pkg unicycle_bringup.launch \
    estimator_type:=dead_reckoning \
    noise_injection:=true \
    freeze_bearing:=false
```

To run `KalmanFilter` estimator:
```angular2html
roslaunch proj3_pkg unicycle_bringup.launch \
    estimator_type:=kalman_filter \
    noise_injection:=true \
    freeze_bearing:=true
```

To run `ExtendedKalmanFilter` estimator:
```angular2html
roslaunch proj3_pkg unicycle_bringup.launch \
    estimator_type:=extended_kalman_filter \
    noise_injection:=true \
    freeze_bearing:=false
```
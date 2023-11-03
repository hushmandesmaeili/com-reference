import numpy as np
import matplotlib.pyplot as plt
from com_trajectory_generator import COMTrajectoryGenerator
from com_trajectory_plotter import COMTrajectoryPlotter


if __name__ == "__main__":
    plot = True

    """
    TEST 1
    """
    endT = 2.0

   # Define Bezier control points
    P0 = np.array([0, 0, 0])
    P1 = np.array([0.02, 0.35, 0])  # Adjust this control point for ease-in/out
    P2 = np.array([0.5, 0.15, 0])

    Kp = 1.0
    Kd = 0.1

    traj = COMTrajectoryGenerator(endT, P0, P1, P2, Kp, Kd)
    plotter = COMTrajectoryPlotter(traj)

    if (plot):
        plotter.plot_trajectories()

    # Example usage of the compute_desired_acceleration method
    currentTime = 1.0
    c_actual = np.array([0.5, 0.5, 0])
    c_dot_actual = np.array([0.1, 0.1, 0])

    C_ddot_desired = traj.compute_desired_acceleration(currentTime, c_actual, c_dot_actual)

    print(C_ddot_desired)


    """
    TEST 2 with new values
    """
    endT = 3.0

   # Define Bezier control points
    P0 = np.array([0, 0, 0])
    P1 = np.array([0.01, 0.5, 0])  # Adjust this control point for ease-in/out
    P2 = np.array([0.7, 0.1, 0])

    Kp = 1.0
    Kd = 0.4

    traj.recompute_trajectories(endT, P0, P1, P2, Kp, Kd)

    if (plot):
        plotter.plot_trajectories()
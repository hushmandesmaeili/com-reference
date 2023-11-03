# com_trajectory_plotter.py
import numpy as np
import matplotlib.pyplot as plt

class COMTrajectoryPlotter:
    def __init__(self, traj, samples=100):
        self.traj = traj
        self.samples = samples

    def plot_trajectories(self):
        t_values = np.linspace(0, self.traj._endT, self.samples)

        # Calculate position, velocity, and acceleration trajectories
        x_values = []
        y_values = []
        vx_values = []
        vy_values = []
        ax_values = []
        ay_values = []

        for tau in t_values:
            position_at_time = self.traj._compute_position_ref(tau)
            velocity_at_time = self.traj._compute_velocity_ref(tau)  # Pass the control points as a single argument
            acceleration_at_time = self.traj._compute_acceleration_ref(tau)  # Pass the control points as a single argument

            x_values.append(position_at_time[0])
            y_values.append(position_at_time[1])
            vx_values.append(velocity_at_time[0])
            vy_values.append(velocity_at_time[1])
            ax_values.append(acceleration_at_time[0])
            ay_values.append(acceleration_at_time[1])

        # Plot the results
        plt.figure(figsize=(9, 9))

        # Plot x and y position trajectories
        plt.subplot(1, 1, 1)
        plt.plot(*zip(*[self.traj._P0, self.traj._P1, self.traj._P2]), marker='o', linestyle='-')
        plt.plot(np.reshape(x_values, (100, 1)), np.reshape(y_values, (100, 1)))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.xlim(0, 0.8)
        plt.ylim(0, 0.8)
        plt.gca().set_aspect('auto')

        plt.figure(figsize=(9, 9))

        # Plot x and y position trajectories
        plt.subplot(3, 1, 1)
        plt.plot(np.reshape(t_values, (100, 1)), np.reshape(x_values, (100, 1)), label='x-position')
        plt.plot(np.reshape(t_values, (100, 1)), np.reshape(y_values, (100, 1)), label='y-position')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.legend()
        plt.grid()
        plt.gca().set_aspect('auto')

        # Plot x and y velocity trajectories
        plt.subplot(3, 1, 2)
        plt.plot(np.reshape(t_values, (100, 1)), np.reshape(vx_values, (100, 1)), label='x-velocity')
        plt.plot(np.reshape(t_values, (100, 1)), np.reshape(vy_values, (100, 1)), label='y-velocity')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.grid()
        plt.gca().set_aspect('auto')

        # Plot x and y acceleration trajectories
        plt.subplot(3, 1, 3)
        plt.plot(np.reshape(t_values, (100, 1)), np.reshape(ax_values, (100, 1)), label='x-acceleration')
        plt.plot(np.reshape(t_values, (100, 1)), np.reshape(ay_values, (100, 1)), label='y-acceleration')
        plt.xlabel('Time')
        plt.ylabel('Acceleration')
        plt.legend()
        plt.grid()
        plt.gca().set_aspect('auto')

        plt.tight_layout()
        plt.show()


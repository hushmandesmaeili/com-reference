import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class COMTrajectoryGenerator:
    def __init__(self, endT, P0, P1, P2, Kp, Kd):
        # Numerical parameters
        self.endT = endT
        self.P0 = P0
        self.P1 = P1
        self.P2 = P2
        self.Kp = Kp
        self.Kd = Kd

        # Declare symbolic variables and expressions
        self.t_sym = ca.MX.sym('t')

        self.position_expr = None
        self.velocity_expr = None
        self.acceleration_expr = None

        # Create symbolic position, velocity, and acceleration functions
        self.position_func = None
        self.velocity_func = None
        self.acceleration_func = None

        self.compute_trajectories()

    def compute_trajectories(self):
        # Define symbolic expressions for the position, velocity, and acceleration trajectories
        # Your symbolic expressions for position, velocity, and acceleration here
        
        # Adjusted timing profile with parametrize end_time and ease-in/ease-out
        tau = (self.t_sym / self.endT) ** 2 / ((self.t_sym / self.endT) ** 2 + (1 - self.t_sym / self.endT) ** 2)

        # Define the position trajectory equations
        X_traj = (1 - tau) ** 2 * self.P0[0] + 2 * (1 - tau) * tau * self.P1[0] + tau ** 2 * self.P2[0]
        Y_traj = (1 - tau) ** 2 * self.P0[1] + 2 * (1 - tau) * tau * self.P1[1] + tau ** 2 * self.P2[1]

        # Concatenate the position trajectory
        self.position_expr = ca.vertcat(
            X_traj,
            Y_traj,
            0
        )

        # Compute the velocity trajectory
        self.velocity_expr = ca.vertcat(
            ca.jacobian(X_traj, self.t_sym),
            ca.jacobian(Y_traj, self.t_sym),
            0
        )

        # Compute the acceleration trajectory
        self.acceleration_expr = ca.vertcat(
            ca.jacobian(self.velocity_expr[0], self.t_sym),
            ca.jacobian(self.velocity_expr[1], self.t_sym),
            0
        )

        # Create symbolic functions
        self.position_func = ca.Function('position_func', [self.t_sym], [self.position_expr])
        self.velocity_func = ca.Function('velocity_func', [self.t_sym], [self.velocity_expr])
        self.acceleration_func = ca.Function('acceleration_func', [self.t_sym], [self.acceleration_expr])

    def compute_desired_acceleration(self, currentTime, c_actual, c_dot_actual):
        if self.position_func is None:
            self.compute_trajectories()

        # Evaluate the symbolic position and velocity at the current time using instance variables
        c_ref = self._compute_position_ref(currentTime)
        c_dot_ref = self._compute_velocity_ref(currentTime)

        # Compute the desired acceleration based on the control law
        c_ddot_ref = self._compute_acceleration_ref(currentTime)
        C_ddot_desired = self.Kp * (c_ref - c_actual) + self.Kd * (c_dot_ref - c_dot_actual) + c_ddot_ref

        return C_ddot_desired
    
    def _compute_position_ref(self, currentTime):
        c_ref = self.position_func(currentTime)

        return c_ref
    
    def _compute_velocity_ref(self, currentTime):
        c_dot_ref = self.velocity_func(currentTime)

        return c_dot_ref
    
    def _compute_acceleration_ref(self, currentTime):
        c_ddot_ref = self.acceleration_func(currentTime)

        return c_ddot_ref


if __name__ == "__main__":
    plot = True
    # Example usage
    endT = 2.0

   # Define Bezier control points
    P0 = np.array([0, 0])
    P1 = np.array([0.02, 0.35])  # Adjust this control point for ease-in/out
    P2 = np.array([0.5, 0.15])

    Kp = 1.0
    Kd = 0.1

    traj = COMTrajectoryGenerator(endT, P0, P1, P2, Kp, Kd)

    if (plot):

        # Define time values for plotting
        t_values = np.linspace(0, endT, 100)  # Adjust the time range as needed

        # Calculate position, velocity, and acceleration trajectories
        x_values = []
        y_values = []
        vx_values = []
        vy_values = []
        ax_values = []
        ay_values = []

        for tau in t_values:
            position_at_time = traj._compute_position_ref(tau)
            velocity_at_time = traj._compute_velocity_ref(tau)  # Pass the control points as a single argument
            acceleration_at_time = traj._compute_acceleration_ref(tau)  # Pass the control points as a single argument

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
        plt.plot(*zip(*[P0, P1, P2]), marker='o', linestyle='-')
        plt.plot(np.reshape(x_values, (100, 1)), np.reshape(y_values, (100, 1)))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.xlim(0, 0.5)
        plt.ylim(0, 0.5)
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

    # Example usage of the compute_desired_acceleration method
    currentTime = 1.0
    c_actual = np.array([0.5, 0.5, 0])
    c_dot_actual = np.array([0.1, 0.1, 0])

    C_ddot_desired = traj.compute_desired_acceleration(currentTime, c_actual, c_dot_actual)

    print(C_ddot_desired)
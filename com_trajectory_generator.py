import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class COMTrajectoryGenerator:
    def __init__(self, endT, P0, P1, P2, Kp, Kd):
        """
        Initialize the COMTrajectoryGenerator with the provided parameters.

        Args:
            endT: End time for the trajectory.
            P0, P1, P2: Control points for the trajectory.
            Kp: Proportional gain for desired acceleration control law.
            Kd: Derivative gain for desired acceleration control law.
        """
        # Numerical parameters
        self._endT = endT
        self._P0 = P0
        self._P1 = P1
        self._P2 = P2
        self._Kp = Kp
        self._Kd = Kd

        # Declare symbolic variables and expressions
        self._t_sym = ca.MX.sym('t')

        self._position_expr = None
        self._velocity_expr = None
        self._acceleration_expr = None

        # Create symbolic position, velocity, and acceleration functions
        self._position_func = None
        self._velocity_func = None
        self._acceleration_func = None

        self._compute_trajectories()

    def _compute_trajectories(self):
        """
        Compute the symbolic expressions for position, velocity, and acceleration trajectories.
        """
        
        # Adjusted timing profile with parametrize end_time and ease-in/ease-out
        tau = (self._t_sym / self._endT) ** 2 / ((self._t_sym / self._endT) ** 2 + (1 - self._t_sym / self._endT) ** 2)

        # Define the position trajectory equations
        X_traj = (1 - tau) ** 2 * self._P0[0] + 2 * (1 - tau) * tau * self._P1[0] + tau ** 2 * self._P2[0]
        Y_traj = (1 - tau) ** 2 * self._P0[1] + 2 * (1 - tau) * tau * self._P1[1] + tau ** 2 * self._P2[1]

        # Concatenate the position trajectory
        self._position_expr = ca.vertcat(
            X_traj,
            Y_traj,
            0
        )

        # Compute the velocity trajectory
        self._velocity_expr = ca.vertcat(
            ca.jacobian(X_traj, self._t_sym),
            ca.jacobian(Y_traj, self._t_sym),
            0
        )

        # Compute the acceleration trajectory
        self._acceleration_expr = ca.vertcat(
            ca.jacobian(self._velocity_expr[0], self._t_sym),
            ca.jacobian(self._velocity_expr[1], self._t_sym),
            0
        )

        # Create symbolic functions
        self._position_func = ca.Function('position_func', [self._t_sym], [self._position_expr])
        self._velocity_func = ca.Function('velocity_func', [self._t_sym], [self._velocity_expr])
        self._acceleration_func = ca.Function('acceleration_func', [self._t_sym], [self._acceleration_expr])

    def recompute_trajectories(self, endT, P0, P1, P2, Kp, Kd):
        """
        Recompute the trajectories with new parameters.

        Args:
            endT: New end time for the trajectory.
            P0, P1, P2: New control points for the trajectory.
            Kp: New proportional gain for control.
            Kd: New derivative gain for control.
        """
        # New numerical parameters
        self._endT = endT
        self._P0 = P0
        self._P1 = P1
        self._P2 = P2
        self._Kp = Kp
        self._Kd = Kd

        # Recompute trajectories
        self._compute_trajectories()

    def _compute_position_ref(self, currentTime):
        c_ref = self._position_func(currentTime)

        return c_ref
    
    def _compute_velocity_ref(self, currentTime):
        c_dot_ref = self._velocity_func(currentTime)

        return c_dot_ref
    
    def _compute_acceleration_ref(self, currentTime):
        c_ddot_ref = self._acceleration_func(currentTime)

        return c_ddot_ref

    def compute_desired_acceleration(self, currentTime, c_actual, c_dot_actual):
        """
        Compute the desired acceleration based on PD control law with feedforward term.

        Args:
            currentTime: Current time.
            c_actual: Current position.
            c_dot_actual: Current velocity.

        Returns:
            C_ddot_desired: Desired acceleration based on control law.
        """
        if self._position_func is None:
            self._compute_trajectories()

        # Evaluate the symbolic position and velocity at the current time using instance variables
        c_ref = self._compute_position_ref(currentTime)
        c_dot_ref = self._compute_velocity_ref(currentTime)

        # Compute the desired acceleration based on the control law
        c_ddot_ref = self._compute_acceleration_ref(currentTime)
        C_ddot_desired = self._Kp * (c_ref - c_actual) + self._Kd * (c_dot_ref - c_dot_actual) + c_ddot_ref

        return C_ddot_desired

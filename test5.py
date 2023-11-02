import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables and functions
t = ca.MX.sym('t')  # Symbolic time variable
endT = ca.MX.sym('endT')  # Symbolic total time variable

# Define Bezier control points
P0 = np.array([0, 0])
P1 = np.array([0.02, 0.35])  # Adjust this control point for ease-in/out
P2 = np.array([0.5, 0.15])

end_time = 2.0

# Adjusted timing profile with parametrize end_time and ease-in/ease-out
f = (t / endT) ** 2 / ((t / endT) ** 2 + (1 - t / endT) ** 2)

# Define the position trajectory equations
X_new = (1 - f) ** 2 * P0[0] + 2 * (1 - f) * f * P1[0] + f ** 2 * P2[0]
Y_new = (1 - f) ** 2 * P0[1] + 2 * (1 - f) * f * P1[1] + f ** 2 * P2[1]

# Compute the velocity trajectory
P_new = ca.vertcat(
    X_new,
    Y_new
)

# Compute the velocity trajectory
V_new = ca.vertcat(
    ca.jacobian(X_new, t),
    ca.jacobian(Y_new, t)
)

# Compute the acceleration trajectory
A_new = ca.vertcat(
    ca.jacobian(V_new[0], t),
    ca.jacobian(V_new[1], t)
)

kp = ca.MX.sym('kp')  # Symbolic time variable
kd = ca.MX.sym('kd')  # Symbolic total time variable

A_new = ca.vertcat(
    ca.jacobian(V_new[0], t),
    ca.jacobian(V_new[1], t)
)
P_curr = ca.MX.sym('P_curr')  # Symbolic total time variable
V_curr = ca.MX.sym('V_curr')  # Symbolic total time variable
Kp = np.array([0, 0])

C_ddot_des = kp * (P_new - P_curr) + kd * (V_new - V_curr) + A_new



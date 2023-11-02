import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define your control points
P0 = np.array([0, 0])
P1 = np.array([0.2, 0.3])  # Adjust this control point for ease-in/out
P2 = np.array([0.75, 0])

# Define the time variable
t = ca.MX.sym('t')

# Bezier curve formula
B_t = (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

# Create a CasADi function to evaluate the Bezier curve
bezier_function = ca.Function('bezier', [t], [B_t])

# Define the number of integration steps
num_steps = 100

# Numerical integration to obtain position
t_values = np.linspace(0, 1, num_steps)
position_values = np.zeros((num_steps, 2))

for i, t_val in enumerate(t_values):
    position_val = bezier_function(t_val)
    position_values[i, 0] = position_val[0, 0]
    position_values[i, 1] = position_val[1, 0]

# Define a function for ease-in ease-out timing profile
def ease_in_out(t):
    return t ** 2 / (t ** 2 + (1 - t) ** 2)

# Apply ease-in ease-out timing to the time values
new_t_values = ease_in_out(t_values)

# Interpolate the position values with the new time values
interpolated_position_values = np.zeros((num_steps, 2))

for i, new_t_val in enumerate(new_t_values):
    interpolated_position_values[i, 0] = np.interp(new_t_val, t_values, position_values[:, 0])
    interpolated_position_values[i, 1] = np.interp(new_t_val, t_values, position_values[:, 1])

# Plot the original and retimed position trajectories
plt.figure(figsize=(12, 6))

# Original position trajectory
plt.subplot(121)
plt.plot(t_values, position_values[:, 0], linestyle='-', label='x-position')
plt.plot(t_values, position_values[:, 1], linestyle='-', label='y-position')
plt.title('Original Position Trajectory')
plt.xlabel('t')
plt.ylabel('Position')
plt.legend(loc='upper right')
plt.gca().set_aspect('equal')

# Retimed position trajectory
plt.subplot(122)
plt.plot(t_values, interpolated_position_values[:, 0], linestyle='-', label='x-position')
plt.plot(t_values, interpolated_position_values[:, 1], linestyle='-', label='y-position')
plt.title('Retimed Position Trajectory')
plt.xlabel('t')
plt.ylabel('Position')
plt.legend(loc='upper right')
plt.gca().set_aspect('equal')

plt.show()

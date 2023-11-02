import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define your control points
P0 = np.array([0, 0])
P1 = np.array([0.2, 0.3])  # Adjust this control point for ease-in/out
P2 = np.array([1.5, 0])

# Define total time
end_time = 1

# Define the time variable
t = ca.MX.sym('t')
endT = ca.MX.sym('endT')

# Bezier curve formula
B_t = (1 - t / endT) ** 2 * P0 + 2 * (1 - t / endT) * (t / endT)* P1 + (t / endT) ** 2 * P2

# Create a CasADi function to evaluate the Bezier curve
bezier_function = ca.Function('bezier', [t, endT], [B_t])

# Define the number of integration steps
num_steps = 100

# Numerical integration to obtain position
t_values = np.linspace(0, end_time, num_steps)
position_values = np.zeros((num_steps, 2))

for i, t_val in enumerate(t_values):
    position_val = bezier_function(t_val, end_time)
    position_values[i, 0] = position_val[0, 0]
    position_values[i, 1] = position_val[1, 0]

# Define a function for ease-in ease-out timing profile
def ease_in_out(t):
    return t ** 2 / (t ** 2 + (1 - t) ** 2)
    # return t * t * (3.0 - 2.0 * t)

# Apply ease-in ease-out timing to the time values
new_t_values = ease_in_out(t_values)

# Interpolate the position values with the new time values
interpolated_position_values = np.zeros((num_steps, 2))

for i, new_t_val in enumerate(new_t_values):
    interpolated_position_values[i, 0] = np.interp(new_t_val, t_values, position_values[:, 0])
    interpolated_position_values[i, 1] = np.interp(new_t_val, t_values, position_values[:, 1])

# # Compute velocity and acceleration values
# delta_t = np.diff(t_values)
# velocity_values = np.gradient(position_values, delta_t)
# acceleration_values = np.gradient(velocity_values, delta_t)

# interpolated_vel_values = np.zeros((num_steps, 2))
# interpolated_acc_values = np.zeros((num_steps, 2))


# Plot the Bezier curve and position trajectory
plt.figure(figsize=(12, 6))

plt.subplot(331)
plt.plot(*zip(*[P0, P1, P2]), marker='o', linestyle='-')
plt.plot(position_values[:, 0], position_values[:, 1], linestyle='-', label='Position Trajectory')
plt.title('Quadratic Bezier Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('auto')

# Original position trajectory
plt.subplot(334)
plt.plot(t_values, position_values[:, 0], linestyle='-', label='x-position')
plt.plot(t_values, position_values[:, 1], linestyle='-', label='y-position')
plt.title('Original Position Trajectory')
plt.xlabel('t')
plt.ylabel('Position')
plt.legend(loc='upper right')
plt.gca().set_aspect('auto')

# # Original velocity trajectory
# plt.subplot(335)
# plt.plot(t_values, velocity_values[:, 0], linestyle='-', label='x-vel')
# plt.plot(t_values, velocity_values[:, 1], linestyle='-', label='y-vel')
# plt.title('Original Velocity Trajectory')
# plt.xlabel('t')
# plt.ylabel('Velocity')
# plt.legend(loc='upper right')
# plt.gca().set_aspect('auto')

# # Original acceleration trajectory
# plt.subplot(336)
# plt.plot(t_values, acceleration_values[:, 0], linestyle='-', label='x-acc')
# plt.plot(t_values, acceleration_values[:, 1], linestyle='-', label='y-acc')
# plt.title('Original Acceleration Trajectory')
# plt.xlabel('t')
# plt.ylabel('Acceleration')
# plt.legend(loc='upper right')
# plt.gca().set_aspect('auto')

# Retimed position trajectory
plt.subplot(337)
plt.plot(t_values, interpolated_position_values[:, 0], linestyle='-', label='x-position')
plt.plot(t_values, interpolated_position_values[:, 1], linestyle='-', label='y-position')
plt.title('Retimed Position Trajectory')
plt.xlabel('t')
plt.ylabel('Position')
plt.legend(loc='upper right')
plt.gca().set_aspect('auto')

plt.show()

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
V_new = ca.vertcat(
    ca.jacobian(X_new, t),
    ca.jacobian(Y_new, t)
)

# Compute the acceleration trajectory
A_new = ca.vertcat(
    ca.jacobian(V_new[0], t),
    ca.jacobian(V_new[1], t)
)

# Create CasADi functions to evaluate velocity and acceleration at a given time
position_function = ca.Function('position_function', [t, endT], [X_new, Y_new])
velocity_function = ca.Function('velocity_function', [t, endT], [V_new])
acceleration_function = ca.Function('acceleration_function', [t, endT], [A_new])

# Define time values for plotting
t_values = np.linspace(0, end_time, 100)  # Adjust the time range as needed

# Calculate position, velocity, and acceleration trajectories
x_values = []
y_values = []
vx_values = []
vy_values = []
ax_values = []
ay_values = []

position = [P0, P1, P2]  # Pass the control points as a list
for tau in t_values:
    position_at_time = position_function(tau, end_time)
    velocity_at_time = velocity_function(tau, end_time)  # Pass the control points as a single argument
    acceleration_at_time = acceleration_function(tau, end_time)  # Pass the control points as a single argument

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

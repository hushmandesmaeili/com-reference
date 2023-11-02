import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define Bezier control points
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

# Define the ease-in ease-out timing function
# timing_function = t**2 / (t**2 + (1 - t)**2)
timing_function = t * t * (3.0 - 2.0 * t)

# Define the retimed parameter tau
tau = timing_function * end_time

# Differentiate tau with respect to t
dtau_dt = ca.jacobian(tau, t)

# Differentiate dtau_dt to find the second derivative
ddtau_dt2 = ca.jacobian(dtau_dt, t)

# Evaluate the second derivative of the Bezier curve at a specific time
# t_value = 0.1  # Change this to the desired time
B_second_derivative = ddtau_dt2
# result = ca.substitute(B_second_derivative, t, t_value)
# result = ca.substitute(B_t, t, t_value)
# result = ca.substitute(result, endT, end_time)

# print(f"Second derivative of the Bezier curve at t = {t_value}: {result}")

# Create functions to evaluate tau, dtau_dt, and ddtau_dt2
evaluate_tau = ca.Function('evaluate_tau', [t], [tau])
evaluate_dtau_dt = ca.Function('evaluate_dtau_dt', [t], [dtau_dt])
evaluate_ddtau_dt2 = ca.Function('evaluate_ddtau_dt2', [t], [ddtau_dt2])

# Generate values of t for plotting
t_values = np.linspace(0, end_time, 100)

# Evaluate the functions for the given t values
tau_values = [evaluate_tau(t_val) for t_val in t_values]
dtau_dt_values = [evaluate_dtau_dt(t_val) for t_val in t_values]
ddtau_dt2_values = [evaluate_ddtau_dt2(t_val) for t_val in t_values]

# Evaluate the Bezier curve for tau values
x_values = [bezier_function(t_val, end_time)[0] for t_val in tau_values]
y_values = [bezier_function(t_val, end_time)[1] for t_val in tau_values]

# Calculate velocities using derivatives
vx_values = [bezier_function(t_val, end_time)[0] * dtau_dt_values[i] for i, t_val in enumerate(tau_values)]
vy_values = [bezier_function(t_val, end_time)[1] * dtau_dt_values[i] for i, t_val in enumerate(tau_values)]

# Calculate accelerations using derivatives
ax_values = [bezier_function(t_val, end_time)[0] * ddtau_dt2_values[i] + 2 * vx_values[i] * dtau_dt_values[i] for i, t_val in enumerate(tau_values)]
ay_values = [bezier_function(t_val, end_time)[1] * ddtau_dt2_values[i] + 2 * vy_values[i] * dtau_dt_values[i] for i, t_val in enumerate(tau_values)]

# Plot the results
plt.figure(figsize=(12, 12))

# Plot x and y position trajectories
plt.subplot(1, 1, 1)
plt.plot(t_values, x_values, label='x-position')
plt.plot(t_values, y_values, label='y-position')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()

# Plot x and y velocity trajectories
plt.subplot(3, 1, 2)
plt.plot(t_values, vx_values, label='x-velocity')
plt.plot(t_values, vy_values, label='y-velocity')
plt.xlabel('Time')
plt.ylabel('Velocity')

# Plot x and y acceleration trajectories
plt.subplot(3, 1, 3)
plt.plot(t_values, ax_values, label='x-acceleration')
plt.plot(t_values, ay_values, label='y-acceleration')
plt.xlabel('tau')
plt.ylabel('Acceleration')
plt.legend()

plt.tight_layout()
plt.grid()
plt.show()
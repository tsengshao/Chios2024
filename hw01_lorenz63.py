import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 10
beta = 8 / 3
rho = 24.74

# Lorenz system equations
def lorenz_system(state, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# Euler method
def euler_step(state, dt, sigma, beta, rho):
    return state + dt * lorenz_system(state, sigma, beta, rho)

# RK4 method
def rk4_step(state, dt, sigma, beta, rho):
    k1 = dt * lorenz_system(state, sigma, beta, rho)
    k2 = dt * lorenz_system(state + k1 / 2, sigma, beta, rho)
    k3 = dt * lorenz_system(state + k2 / 2, sigma, beta, rho)
    k4 = dt * lorenz_system(state + k3, sigma, beta, rho)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Time settings
dt_euler = 0.01  # Chosen time step for Euler method (small for stability)
dt_rk4 = 0.05  # Larger time step for RK4 due to its accuracy
t_max = 40
num_steps_euler = int(t_max / dt_euler)
num_steps_rk4 = int(t_max / dt_rk4)

# Initial conditions
initial_state = np.array([0.1, 0, 0])

# Running the Euler method
states_euler = np.zeros((num_steps_euler, 3))
states_euler[0] = initial_state
for i in range(1, num_steps_euler):
    states_euler[i] = euler_step(states_euler[i - 1], dt_euler, sigma, beta, rho)

# Running the RK4 method
states_rk4 = np.zeros((num_steps_rk4, 3))
states_rk4[0] = initial_state
for i in range(1, num_steps_rk4):
    states_rk4[i] = rk4_step(states_rk4[i - 1], dt_rk4, sigma, beta, rho)

# Time arrays
time_euler = np.linspace(0, t_max, num_steps_euler)
time_rk4 = np.linspace(0, t_max, num_steps_rk4)

# Plotting the results
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# Euler method plots
axs[0, 0].plot(time_euler, states_euler[:, 0])
axs[0, 0].set_title('Euler Method: x vs time')
axs[1, 0].plot(time_euler, states_euler[:, 1])
axs[1, 0].set_title('Euler Method: y vs time')
axs[2, 0].plot(time_euler, states_euler[:, 2])
axs[2, 0].set_title('Euler Method: z vs time')

# RK4 method plots
axs[0, 1].plot(time_rk4, states_rk4[:, 0])
axs[0, 1].set_title('RK4 Method: x vs time')
axs[1, 1].plot(time_rk4, states_rk4[:, 1])
axs[1, 1].set_title('RK4 Method: y vs time')
axs[2, 1].plot(time_rk4, states_rk4[:, 2])
axs[2, 1].set_title('RK4 Method: z vs time')

# Final adjustments
for ax in axs.flat:
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

plt.tight_layout()
plt.show()


# 3D Plot of the Lorenz system to show the butterfly effect

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory from RK4 (since it's more accurate)
ax.plot(states_rk4[:, 0], states_rk4[:, 1], states_rk4[:, 2], color='b')

# Labels and title
ax.set_title('Butterfly Effect (Lorenz System) - RK4 Method')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()



import numpy as np
import matplotlib.pyplot as plt
import sys, os
import numba

# Parameters
sigma = 10
beta = 8 / 3
rho = 24.74

# Lorenz system equations
## @numba.guvectorize([(float64[:], float64, float64, float64)], '(n),(),(),()->(n)')
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
dt_euler = 0.0000001  # Chosen time step for Euler method (small for stability)
dt_rk4 = 0.0000001  # Larger time step for RK4 due to its accuracy
t_max = 40
num_steps_euler = int(t_max / dt_euler)
num_steps_rk4 = int(t_max / dt_rk4)

# Initial conditions
dum=np.sqrt(beta*(rho-1))
initial_state = np.array([0, 0, 0], dtype=float)
#initial_state = np.array([dum, dum, rho-1])
#initial_state = np.array([-dum, -dum, rho-1])

#add perturbation
initial_state[0] = initial_state[0]+0.1

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

# save results
c = np.concatenate([time_euler[:,np.newaxis], states_euler], axis=1)
np.savetxt(f'{dt_euler}_euler.txt',c,header='ts x y z')

c = np.concatenate([time_rk4[:,np.newaxis], states_rk4], axis=1)
np.savetxt(f'{dt_rk4}_rk4.txt',c,header='ts x y z')


####################################
# Plotting the results
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

# find ylim for x, y, z
ylim_min = np.vstack((np.min(states_euler, axis=0),\
                      np.min(states_rk4, axis=0),\
                    )).min(axis=0)
ylim_max = np.vstack((np.max(states_euler, axis=0),\
                      np.max(states_rk4, axis=0),\
                    )).max(axis=0)

# Euler method plots
axis_name=['x','y','z']
for i in range(3):
  axs[i, 0].plot(time_euler, states_euler[:, i])
  axs[i, 0].set_title(f'Euler Method: {axis_name[i]} vs time')
  axs[i, 0].set_ylim(ylim_min[i], ylim_max[i])

# RK4 method plots
for i in range(3):
  axs[i, 1].plot(time_rk4, states_rk4[:, i])
  axs[i, 1].set_title(f'RK4 Method: {axis_name[i]} vs time')
  axs[i, 1].set_ylim(ylim_min[i], ylim_max[i])

# Final adjustments
for ax in axs.flat:
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

plt.tight_layout()
plt.show()


####################################
# Plotting the results in same figure
fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
# find ylim for x, y, z
ylim_min = np.vstack((np.min(states_euler, axis=0),\
                      np.min(states_rk4, axis=0),\
                    )).min(axis=0)
ylim_max = np.vstack((np.max(states_euler, axis=0),\
                      np.max(states_rk4, axis=0),\
                    )).max(axis=0)
# Euler method plots
axis_name=['x','y','z']
for i in range(3):
  axs[i].plot(time_euler, states_euler[:, i], label='Euler')
  axs[i].plot(time_rk4,   states_rk4[:, i], label='RK4')
  axs[i].set_title(f'{axis_name[i]} vs time')
  axs[i].set_ylim(ylim_min[i], ylim_max[i])
  axs[i].legend()

plt.show()


# 3D Plot of the Lorenz system to show the butterfly effect

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory from RK4 (since it's more accurate)
ax.plot(states_euler[:, 0], \
        states_euler[:, 1], \
        states_euler[:, 2], \
        color='C0',label='Euler')
ax.plot(states_rk4[:, 0], \
        states_rk4[:, 1], \
        states_rk4[:, 2], \
        color='C1',label='RK4')
plt.legend()

# Labels and title
ax.set_title('Butterfly Effect (Lorenz System) - Euler/RK4 Method')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()



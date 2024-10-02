import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.integrate import solve_ivp
from scipy.linalg    import eigvals

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
    output = np.array([dxdt, dydt, dzdt])
    return output

def asymptotic_lorenz_system(state):
    u, v, w = state
    dudt = u
    dvdt = -1 * u * w
    dwdt = u * v
    return np.array([dudt, dvdt, dwdt])

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

def run_lorenz_experiment(method, dt, t_max, initial_state, sigma, beta, rho):
    #print(method, dt, t_max, initial_state, sigma, beta, rho)
    time = np.arange(0, t_max+dt, dt)
    num_steps = time.size
    # Running the Euler method
    states = np.zeros((num_steps, 3))
    states[0] = initial_state

    if method=='sci':
        # solve from scipy
        warpper_func = lambda t, y: lorenz_system(y, sigma, beta, rho)
        result_sci = solve_ivp(fun       = warpper_func, \
                               t_span    = (0,t_max), \
                               y0        = initial_state, \
                               t_eval    = time, \
                              )
        states = result_sci.y.T
        time   = result_sci.t
        return time, states
    elif method=='rk4':
        func = rk4_step
    elif method=='euler':
        func = euler_step

    for i in range(1, num_steps):
        states[i] = func(states[i - 1], dt, sigma, beta, rho)
    return time, states
    

# Function to run the Lorenz system with a given time step and method, checking for stability
def determine_longest_stable_step(method, initial_state, t_max, sigma, beta, rho, step_sizes):
    longest_stable_step = None
    for dt in step_sizes:
        try:
            ti, states = run_lorenz_experiment(method, dt, t_max, initial_state, sigma, beta, rho)
            # Check for NaNs or excessively large values (divergence)
            if not np.isnan(states).any() and np.abs(states).max() < 1e5:
                longest_stable_step = dt
            else:
                # If the solution becomes unstable, break the loop
                break
        except Exception as e:
            # Handle any other potential errors (overflow, etc.)
            break
    return longest_stable_step

# solve jac
jac = [ [-sigma, sigma, 0], \
        [1, -1, -np.sqrt(beta*(rho-1))], \
        [np.sqrt(beta*(rho-1)), np.sqrt(beta*(rho-1)), -beta],\
      ]
evalue = eigvals(jac)
print('jac:')
print(np.array(jac))
print('the eigen value is :',evalue)

# Time settings
dt_euler = 0.001 # Chosen time step for Euler method (small for stability)
dt_rk4 = 0.001  # Larger time step for RK4 due to its accuracy
dt_sci = 0.01  # Larger time step for RK4 due to its accuracy
t_max = 40

# Initial conditions
dum=np.sqrt(beta*(rho-1))
initial_state = np.array([0, 0, 0], dtype=float)
#initial_state = np.array([dum, dum, rho-1])
initial_state[0] = initial_state[0]+0.1

args = {'t_max':t_max, \
        'initial_state':initial_state, \
        'sigma':sigma, \
        'beta':beta, \
        'rho':rho,\
       }

time_euler, states_euler = run_lorenz_experiment('euler', dt_euler, **args)
time_rk4,   states_rk4   = run_lorenz_experiment('rk4',   dt_rk4,   **args)
time_sci,   states_sci   = run_lorenz_experiment('sci',   dt_sci,   **args)

epsilon = rho**(-0.5)
tau = time_euler/epsilon

# test stable time step
# Step sizes for the experiments
step_sizes_euler = np.linspace(0.001, 0.05, 50)  # Finer for Euler due to its lower stability
step_sizes_rk4 = np.linspace(0.01, 0.2, 50)  # Larger range for RK4

longest_stable_step_euler = \
    determine_longest_stable_step('euler', step_sizes=step_sizes_euler, **args)

longest_stable_step_rk4 = \
    determine_longest_stable_step('rk4', step_sizes=step_sizes_rk4, **args)
print('-------------------------------------')
print('the longest timestep ...')
print(f'Euler method: {longest_stable_step_euler:.2e}')
print(f'RK4   method: {longest_stable_step_rk4:.2e}')


####################################
# Plotting the results
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

# find ylim for x, y, z
ylim_min = np.nanmin(\
                      np.vstack((np.min(states_euler, axis=0),\
                      np.min(states_rk4, axis=0),\
                    )), axis=0)
ylim_max = np.nanmax(\
                     np.vstack((np.max(states_euler, axis=0),\
                     np.max(states_rk4, axis=0),\
                    )), axis=0)

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
# Euler method plots
axis_name=['x','y','z']
for i in range(3):
  axs[i].plot(time_sci, states_sci[:, i], label='Sci')
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
ax.plot(states_sci[:, 0], \
        states_sci[:, 1], \
        states_sci[:, 2], \
        color='C0',label='sci')
ax.plot(states_euler[:, 0], \
        states_euler[:, 1], \
        states_euler[:, 2], \
        color='C1',label='Euler')
ax.plot(states_rk4[:, 0], \
        states_rk4[:, 1], \
        states_rk4[:, 2], \
        color='C2',label='RK4')
plt.legend()

# Labels and title
ax.set_title('Butterfly Effect (Lorenz System) - Euler/RK4 Method')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()



import numpy as np
import matplotlib.pyplot as plt


def serid_rk4(s0, e0, r0, i0, d0, beta, gamma, sigma, mu, t_max, dt):
    num_steps = int(t_max/dt)
    t = np.linspace(0, t_max, num_steps)
    s = np.zeros(num_steps)
    e = np.zeros(num_steps)
    r = np.zeros(num_steps)
    i = np.zeros(num_steps)
    d = np.zeros(num_steps)
    s[0] = s0
    e[0] = e0
    r[0] = r0
    i[0] = i0
    d[0] = d0
    
    for n in range(1, num_steps):
        s_n = s[n-1]
        e_n = e[n-1]
        r_n = r[n-1]
        i_n = i[n-1]
        d_n = d[n-1]
        
        # Compute the derivatives
        ds_dt = mu - beta * s_n * i_n - mu * s_n
        de_dt = beta * s_n * i_n - sigma * e_n - mu * e_n
        dr_dt = gamma * i_n - mu * r_n
        di_dt = sigma * e_n - gamma * i_n - mu * i_n
        dd_dt = mu * (s_n + e_n + r_n + i_n)
        
        # Update the state variables using the Runge-Kutta method
        s[n] = s_n + dt * ds_dt
        e[n] = e_n + dt * de_dt
        r[n] = r_n + dt * dr_dt
        i[n] = i_n + dt * di_dt
        d[n] = d_n + dt * dd_dt
    
    return t, s, e, r, i, d


t_max = 100
dt = 0.1

s0 = 1000
e0 = 10
r0 = 0
i0 = 5
d0 = 0
beta = 0.1
gamma = 0.04
sigma = 0.05
mu = 0.005


# Run the simulation
t, s, e, r, i, d = serid_rk4(s0, e0, r0, i0, d0, beta, gamma, sigma, mu, t_max, dt)


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, s, label='Susceptible')
plt.plot(t, e, label='Exposed')
plt.plot(t, r, label='Recovered')
plt.plot(t, i, label='Infected')
plt.plot(t, d, label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SERID Model Simulation')
plt.legend()
plt.grid()
plt.show()

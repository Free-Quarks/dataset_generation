import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, n_population, i_initial, t_max, dt):
    # Define the SIR model
    def model(state, t):
        s, i, r = state
        ds_dt = -beta * s * i / n_population
        di_dt = beta * s * i / n_population - gamma * i
        dr_dt = gamma * i
        return ds_dt, di_dt, dr_dt
    
    # Initialize arrays
    num_steps = int(t_max / dt)
    t = np.linspace(0, t_max, num_steps)
    s = np.zeros(num_steps)
    i = np.zeros(num_steps)
    r = np.zeros(num_steps)
    
    # Set initial conditions
    s[0] = n_population - i_initial
    i[0] = i_initial
    r[0] = 0
    
    # Solve using RK4
    for step in range(1, num_steps):
        state = s[step-1], i[step-1], r[step-1]
        k1 = model(state, t[step-1])
        k2 = model(state + 0.5*dt*k1, t[step-1] + 0.5*dt)
        k3 = model(state + 0.5*dt*k2, t[step-1] + 0.5*dt)
        k4 = model(state + dt*k3, t[step-1] + dt)
        s[step] = s[step-1] + (1 / 6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        i[step] = i[step-1] + (1 / 6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        r[step] = r[step-1] + (1 / 6) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    
    # Plot the results
    plt.plot(t, s, label='Susceptible')
    plt.plot(t, i, label='Infected')
    plt.plot(t, r, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of People')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.show()

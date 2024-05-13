import numpy as np
import matplotlib.pyplot as plt


# Function to implement the epidemiology compartmental model

def serid_model(t, x, beta, gamma, delta):
    s, e, r, i, d = x
    ds_dt = -beta * s * i
    de_dt = beta * s * i - gamma * e - delta * e
    dr_dt = gamma * e
    di_dt = delta * e - r
    dd_dt = delta * e
    return [ds_dt, de_dt, dr_dt, di_dt, dd_dt]


# Function to simulate and plot the model

def simulate_serid_model(beta, gamma, delta, initial_conditions, t_start, t_end, num_points):
    t = np.linspace(t_start, t_end, num_points)
    s_0, e_0, r_0, i_0, d_0 = initial_conditions
    x_0 = [s_0, e_0, r_0, i_0, d_0]
    result = np.zeros((num_points, len(x_0)))
    result[0] = x_0
    h = (t_end - t_start) / num_points
    for i in range(num_points - 1):
        k1 = h * np.array(serid_model(t[i], result[i], beta, gamma, delta))
        k2 = h * np.array(serid_model(t[i] + h/2, result[i] + k1/2, beta, gamma, delta))
        k3 = h * np.array(serid_model(t[i] + h/2, result[i] + k2/2, beta, gamma, delta))
        k4 = h * np.array(serid_model(t[i] + h, result[i] + k3, beta, gamma, delta))
        result[i+1] = result[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    plt.figure()
    plt.plot(t, result[:, 0], label='Susceptible')
    plt.plot(t, result[:, 1], label='Exposed')
    plt.plot(t, result[:, 2], label='Recovered')
    plt.plot(t, result[:, 3], label='Infected')
    plt.plot(t, result[:, 4], label='Deceased')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()


# Example usage

beta = 0.2
gamma = 0.1
delta = 0.05
initial_conditions = [1000, 10, 0, 1, 0]
t_start = 0
t_end = 100
num_points = 1000

simulate_serid_model(beta, gamma, delta, initial_conditions, t_start, t_end, num_points)

import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, S0, I0, R0, t_max, dt):
    # Define initial conditions
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = [0]
    
    # Define the differential equations
    def ds_dt(S, I):
        return -beta * S * I / N
    
    def di_dt(S, I):
        return beta * S * I / N - gamma * I
    
    def dr_dt(I):
        return gamma * I
    
    # Implement the Runge-Kutta method
    while t[-1] < t_max:
        h = min(dt, t_max - t[-1])
        k1_s = ds_dt(S[-1], I[-1])
        k1_i = di_dt(S[-1], I[-1])
        k1_r = dr_dt(I[-1])
        k2_s = ds_dt(S[-1] + h / 2 * k1_s, I[-1] + h / 2 * k1_i)
        k2_i = di_dt(S[-1] + h / 2 * k1_s, I[-1] + h / 2 * k1_i)
        k2_r = dr_dt(I[-1] + h / 2 * k1_i)
        k3_s = ds_dt(S[-1] + h / 2 * k2_s, I[-1] + h / 2 * k2_i)
        k3_i = di_dt(S[-1] + h / 2 * k2_s, I[-1] + h / 2 * k2_i)
        k3_r = dr_dt(I[-1] + h / 2 * k2_i)
        k4_s = ds_dt(S[-1] + h * k3_s, I[-1] + h * k3_i)
        k4_i = di_dt(S[-1] + h * k3_s, I[-1] + h * k3_i)
        k4_r = dr_dt(I[-1] + h * k3_i)
        S.append(S[-1] + h / 6 * (k1_s + 2 * k2_s + 2 * k3_s + k4_s))
        I.append(I[-1] + h / 6 * (k1_i + 2 * k2_i + 2 * k3_i + k4_i))
        R.append(R[-1] + h / 6 * (k1_r + 2 * k2_r + 2 * k3_r + k4_r))
        t.append(t[-1] + h)
    
    return S, I, R


# Example usage

# Define parameters
beta = 0.3
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_max = 100
dt = 0.1

# Run the simulation
S, I, R = sir_model(beta, gamma, S0, I0, R0, t_max, dt)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()


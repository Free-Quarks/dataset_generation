import numpy as np
import matplotlib.pyplot as plt


# Function to implement the SIR model

def SIR_model(beta, gamma, population, I0, T):
    N = population
    S0 = N - I0
    R0 = 0
    I = I0
    S = S0
    R = R0
    dt = 0.1
    t = np.arange(0, T+dt, dt)
    S_arr = np.zeros(len(t))
    I_arr = np.zeros(len(t))
    R_arr = np.zeros(len(t))
    S_arr[0] = S
    I_arr[0] = I
    R_arr[0] = R
    for i in range(1, len(t)):
        dS_dt = -beta * S * I / N
        dI_dt = beta * S * I / N - gamma * I
        dR_dt = gamma * I
        S = S + dt * dS_dt
        I = I + dt * dI_dt
        R = R + dt * dR_dt
        S_arr[i] = S
        I_arr[i] = I
        R_arr[i] = R
    return t, S_arr, I_arr, R_arr


# Parameters
beta = 0.3
gamma = 0.1
population = 1000
I0 = 1
T = 100

# Calling the SIR model function
t, S, I, R = SIR_model(beta, gamma, population, I0, T)

# Plotting the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

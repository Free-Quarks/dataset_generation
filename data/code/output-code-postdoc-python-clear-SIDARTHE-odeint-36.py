import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Function that defines the system of differential equations

def SIDARTHE(y, t, beta, gamma, delta, theta, alpha, rho):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    
    # Define the differential equations for each compartment
    dSdt = -beta * S * (I + theta * A + alpha * T) / N
    dIdt = (beta * S * (I + theta * A + alpha * T) / N) - (gamma + delta + rho) * I
    dDdt = delta * I
    dAdt = rho * I - (gamma + theta) * A
    dRdt = gamma * (I + A)
    dTdt = theta * (I + A) - alpha * T
    dHdt = alpha * T
    dEdt = theta * A
    
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


# Function to run simulation and generate plots

def run_simulation(S0, I0, D0, A0, R0, T0, H0, E0, beta, gamma, delta, theta, alpha, rho, timesteps):
    # Set initial conditions
    y0 = [S0, I0, D0, A0, R0, T0, H0, E0]

    # Set time points to evaluate the ODEs
    t = np.linspace(0, timesteps, num=timesteps+1)

    # Solve the ODEs
    sol = odeint(SIDARTHE, y0, t, args=(beta, gamma, delta, theta, alpha, rho))

    # Plot the results
    plt.plot(t, sol[:, 0], label='S')
    plt.plot(t, sol[:, 1], label='I')
    plt.plot(t, sol[:, 2], label='D')
    plt.plot(t, sol[:, 3], label='A')
    plt.plot(t, sol[:, 4], label='R')
    plt.plot(t, sol[:, 5], label='T')
    plt.plot(t, sol[:, 6], label='H')
    plt.plot(t, sol[:, 7], label='E')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()


# Example usage:

# run_simulation(S0, I0, D0, A0, R0, T0, H0, E0, beta, gamma, delta, theta, alpha, rho, timesteps)

# Set initial conditions
S0 = 1000000
I0 = 1000
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0

# Set model parameters
beta = 0.2
gamma = 0.1
delta = 0.01
theta = 0.05
alpha = 0.02ho = 0.01

timesteps = 100

# Run simulation
run_simulation(S0, I0, D0, A0, R0, T0, H0, E0, beta, gamma, delta, theta, alpha, rho, timesteps)

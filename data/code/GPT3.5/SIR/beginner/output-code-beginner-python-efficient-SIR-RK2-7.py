import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(S0, I0, R0, beta, gamma, t_max, dt):
    # Initialize arrays
    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0

    # Iterate over time steps
    for i in range(1, len(t)):
        # Calculate changes in each compartment
        dS = -beta * S[i-1] * I[i-1] * dt
        dI = (beta * S[i-1] * I[i-1] - gamma * I[i-1]) * dt
        dR = gamma * I[i-1] * dt

        # Update compartments using RK2 method
        S_k1 = S[i-1] + dS
        I_k1 = I[i-1] + dI
        R_k1 = R[i-1] + dR

        S_k2 = S[i-1] + 0.5 * dS
        I_k2 = I[i-1] + 0.5 * dI
        R_k2 = R[i-1] + 0.5 * dR

        S[i] = S[i-1] + 0.5 * (dS + (-beta * S_k2 * I_k2 * dt))
        I[i] = I[i-1] + 0.5 * (dI + (beta * S_k2 * I_k2 - gamma * I_k2) * dt)
        R[i] = R[i-1] + 0.5 * (dR + (gamma * I_k2) * dt)

    # Return the arrays of compartments
    return S, I, R


# Example usage
S0 = 1000  # Initial susceptible population
I0 = 1     # Initial infected population
R0 = 0     # Initial recovered population
beta = 0.2 # Infection rate
gamma = 0.1 # Recovery rate
t_max = 100 # Maximum time
dt = 0.1    # Time step

S, I, R = SIR_RK2(S0, I0, R0, beta, gamma, t_max, dt)

# Plotting
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()


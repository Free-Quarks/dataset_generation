import numpy as np


def SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt):
    """
    Simulate the SIR model using 4th order Runge-Kutta method.
    """
    # Define the differential equations
    def dS(S, I, R):
        return -beta * S * I

    def dI(S, I, R):
        return beta * S * I - gamma * I

    def dR(S, I, R):
        return gamma * I

    # Initialize arrays to store the values
    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0

    # Apply the 4th order Runge-Kutta method
    for i in range(1, len(t)):
        k1_S = dt * dS(S[i-1], I[i-1], R[i-1])
        k1_I = dt * dI(S[i-1], I[i-1], R[i-1])
        k1_R = dt * dR(S[i-1], I[i-1], R[i-1])

        k2_S = dt * dS(S[i-1] + 0.5 * k1_S, I[i-1] + 0.5 * k1_I, R[i-1] + 0.5 * k1_R)
        k2_I = dt * dI(S[i-1] + 0.5 * k1_S, I[i-1] + 0.5 * k1_I, R[i-1] + 0.5 * k1_R)
        k2_R = dt * dR(S[i-1] + 0.5 * k1_S, I[i-1] + 0.5 * k1_I, R[i-1] + 0.5 * k1_R)

        k3_S = dt * dS(S[i-1] + 0.5 * k2_S, I[i-1] + 0.5 * k2_I, R[i-1] + 0.5 * k2_R)
        k3_I = dt * dI(S[i-1] + 0.5 * k2_S, I[i-1] + 0.5 * k2_I, R[i-1] + 0.5 * k2_R)
        k3_R = dt * dR(S[i-1] + 0.5 * k2_S, I[i-1] + 0.5 * k2_I, R[i-1] + 0.5 * k2_R)

        k4_S = dt * dS(S[i-1] + k3_S, I[i-1] + k3_I, R[i-1] + k3_R)
        k4_I = dt * dI(S[i-1] + k3_S, I[i-1] + k3_I, R[i-1] + k3_R)
        k4_R = dt * dR(S[i-1] + k3_S, I[i-1] + k3_I, R[i-1] + k3_R)

        S[i] = S[i-1] + (1 / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        I[i] = I[i-1] + (1 / 6) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)
        R[i] = R[i-1] + (1 / 6) * (k1_R + 2 * k2_R + 2 * k3_R + k4_R)

    return t, S, I, R


# Example usage
beta = 0.2
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0.0

t_max = 100
dt = 0.1

t, S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(t, S, label='S')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('SIR Model (RK4)')
plt.legend()
plt.show()

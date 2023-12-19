import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(b, g, S0, I0, R0, t_end, N):
    """
    Simulate SIR model using Runge-Kutta 4th order method
    """
    # Initialize arrays
    S = np.zeros(t_end+1)
    I = np.zeros(t_end+1)
    R = np.zeros(t_end+1)
    t = np.arange(t_end+1)

    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0

    # Calculate step size
    dt = 1

    # Simulate SIR model using RK4
    for i in range(t_end):
        dSdt = -b * S[i] * I[i] / N
        dIdt = (b * S[i] * I[i] / N) - (g * I[i])
        dRdt = g * I[i]

        S_k1 = dt * dSdt
        I_k1 = dt * dIdt
        R_k1 = dt * dRdt

        dSdt = -b * (S[i] + S_k1/2) * (I[i] + I_k1/2) / N
        dIdt = (b * (S[i] + S_k1/2) * (I[i] + I_k1/2) / N) - (g * (I[i] + I_k1/2))
        dRdt = g * (I[i] + I_k1/2)

        S_k2 = dt * dSdt
        I_k2 = dt * dIdt
        R_k2 = dt * dRdt

        dSdt = -b * (S[i] + S_k2/2) * (I[i] + I_k2/2) / N
        dIdt = (b * (S[i] + S_k2/2) * (I[i] + I_k2/2) / N) - (g * (I[i] + I_k2/2))
        dRdt = g * (I[i] + I_k2/2)

        S_k3 = dt * dSdt
        I_k3 = dt * dIdt
        R_k3 = dt * dRdt

        dSdt = -b * (S[i] + S_k3) * (I[i] + I_k3) / N
        dIdt = (b * (S[i] + S_k3) * (I[i] + I_k3) / N) - (g * (I[i] + I_k3))
        dRdt = g * (I[i] + I_k3)

        S_k4 = dt * dSdt
        I_k4 = dt * dIdt
        R_k4 = dt * dRdt

        S[i+1] = S[i] + (S_k1 + 2*S_k2 + 2*S_k3 + S_k4) / 6
        I[i+1] = I[i] + (I_k1 + 2*I_k2 + 2*I_k3 + I_k4) / 6
        R[i+1] = R[i] + (R_k1 + 2*R_k2 + 2*R_k3 + R_k4) / 6

    return S, I, R


# Set parameters
beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
N = S0 + I0 + R0
t_end = 100

# Run simulation
S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, t_end, N)

# Plot results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(beta, gamma, S0, I0, R0, t_end, dt):
    # Define the differential equations
    def dSdt(t, S, I, R):
        return -beta * S * I

    def dIdt(t, S, I, R):
        return beta * S * I - gamma * I

    def dRdt(t, S, I, R):
        return gamma * I

    # Initialize arrays
    t = np.arange(0, t_end, dt)
    num_steps = len(t)
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    # Runge-Kutta 4th order method
    for i in range(1, num_steps):
        k1_S = dSdt(t[i-1], S[i-1], I[i-1], R[i-1])
        k1_I = dIdt(t[i-1], S[i-1], I[i-1], R[i-1])
        k1_R = dRdt(t[i-1], S[i-1], I[i-1], R[i-1])
        k2_S = dSdt(t[i-1] + dt/2, S[i-1] + dt/2 * k1_S, I[i-1] + dt/2 * k1_I, R[i-1] + dt/2 * k1_R)
        k2_I = dIdt(t[i-1] + dt/2, S[i-1] + dt/2 * k1_S, I[i-1] + dt/2 * k1_I, R[i-1] + dt/2 * k1_R)
        k2_R = dRdt(t[i-1] + dt/2, S[i-1] + dt/2 * k1_S, I[i-1] + dt/2 * k1_I, R[i-1] + dt/2 * k1_R)
        k3_S = dSdt(t[i-1] + dt/2, S[i-1] + dt/2 * k2_S, I[i-1] + dt/2 * k2_I, R[i-1] + dt/2 * k2_R)
        k3_I = dIdt(t[i-1] + dt/2, S[i-1] + dt/2 * k2_S, I[i-1] + dt/2 * k2_I, R[i-1] + dt/2 * k2_R)
        k3_R = dRdt(t[i-1] + dt/2, S[i-1] + dt/2 * k2_S, I[i-1] + dt/2 * k2_I, R[i-1] + dt/2 * k2_R)
        k4_S = dSdt(t[i-1] + dt, S[i-1] + dt * k3_S, I[i-1] + dt * k3_I, R[i-1] + dt * k3_R)
        k4_I = dIdt(t[i-1] + dt, S[i-1] + dt * k3_S, I[i-1] + dt * k3_I, R[i-1] + dt * k3_R)
        k4_R = dRdt(t[i-1] + dt, S[i-1] + dt * k3_S, I[i-1] + dt * k3_I, R[i-1] + dt * k3_R)
        S[i] = S[i-1] + dt/6 * (k1_S + 2*k2_S + 2*k3_S + k4_S)
        I[i] = I[i-1] + dt/6 * (k1_I + 2*k2_I + 2*k3_I + k4_I)
        R[i] = R[i-1] + dt/6 * (k1_R + 2*k2_R + 2*k3_R + k4_R)

    return t, S, I, R


# Define parameters
beta = 0.3
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0

# Define simulation time
t_end = 100
dt = 0.1

# Run simulation
t, S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, t_end, dt)

# Plot results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population Fraction')
plt.title('SIR Model using RK4')
plt.legend()
plt.show()

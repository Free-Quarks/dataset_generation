import numpy as np
import matplotlib.pyplot as plt


def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def RK4_integration(y0, t, beta, gamma):
    n = len(t)
    dt = t[1] - t[0]
    S = np.zeros(n)
    I = np.zeros(n)
    R = np.zeros(n)
    S[0], I[0], R[0] = y0
    for i in range(1, n):
        k1 = SIR_model([S[i-1], I[i-1], R[i-1]], t[i-1], beta, gamma)
        k2 = SIR_model([S[i-1] + 0.5*dt*k1[0], I[i-1] + 0.5*dt*k1[1], R[i-1] + 0.5*dt*k1[2]], t[i-1] + 0.5*dt, beta, gamma)
        k3 = SIR_model([S[i-1] + 0.5*dt*k2[0], I[i-1] + 0.5*dt*k2[1], R[i-1] + 0.5*dt*k2[2]], t[i-1] + 0.5*dt, beta, gamma)
        k4 = SIR_model([S[i-1] + dt*k3[0], I[i-1] + dt*k3[1], R[i-1] + dt*k3[2]], t[i], beta, gamma)
        S[i] = S[i-1] + (1/6) * dt * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        I[i] = I[i-1] + (1/6) * dt * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        R[i] = R[i-1] + (1/6) * dt * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    return S, I, R


# Define the initial conditions
y0 = [990, 10, 0]

# Define the time vector
t = np.linspace(0, 100, 1000)

# Define the parameters
beta = 0.3
gamma = 0.1

# Perform the integration using RK4
S, I, R = RK4_integration(y0, t, beta, gamma)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK4')
plt.legend()
plt.show()


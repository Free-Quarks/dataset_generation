import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, T):
    # Step size
    dt = 0.01
    num_steps = int(T/dt)

    # Initialize arrays
    t = np.linspace(0, T, num_steps+1)
    S = np.zeros(num_steps+1)
    I = np.zeros(num_steps+1)
    R = np.zeros(num_steps+1)

    # Set initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0

    # Runge-Kutta method
    for i in range(num_steps):
        dSdt = -beta * S[i] * I[i] / N
        dIdt = beta * S[i] * I[i] / N - gamma * I[i]
        dRdt = gamma * I[i]
        S[i+1] = S[i] + dt * dSdt
        I[i+1] = I[i] + dt * dIdt
        R[i+1] = R[i] + dt * dRdt

    # Return results
    return t, S, I, R

# Parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
T = 30

# Run the model
t, S, I, R = SIR_RK3(beta, gamma, N, I0, T)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, t_max, dt):
    t = np.arange(0, t_max, dt)
    N = S0 + I0 + R0
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        k1 = -beta * S[i-1] * I[i-1] / N
        k2 = -beta * (S[i-1] + dt/2 * k1) * (I[i-1] + dt/2 * k1) / N
        S[i] = S[i-1] + dt * k2
        k1 = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2 = beta * (S[i-1] + dt/2 * k1) * (I[i-1] + dt/2 * k1) / N - gamma * (I[i-1] + dt/2 * k1)
        I[i] = I[i-1] + dt * k2
        k1 = gamma * I[i-1]
        k2 = gamma * (I[i-1] + dt/2 * k1)
        R[i] = R[i-1] + dt * k2
    return t, S, I, R


# example usage
t_max = 100
dt = 0.1
beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0

# Run the model
t, S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, t_max, dt)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model using RK2')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S0, I0, R0, t_max):
    N = S0 + I0 + R0
    dt = 0.01
    t = np.linspace(0, t_max, int(t_max/dt)+1)
    S, I, R = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1]/N
        dI = beta * S[i-1] * I[i-1]/N - gamma * I[i-1]
        dR = gamma * I[i-1]
        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
    return t, S, I, R

beta = 0.2
gamma = 0.1
S0 = 999
I0 = 1
R0 = 0
t_max = 100

# Run SIR model
t, S, I, R = SIR_model(beta, gamma, S0, I0, R0, t_max)

# Plot results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

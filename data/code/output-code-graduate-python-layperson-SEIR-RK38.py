import numpy as np
import matplotlib.pyplot as plt
def seir_model(beta, gamma, sigma, initial_conditions, t_max):
    S_0, E_0, I_0, R_0 = initial_conditions
    N = S_0 + E_0 + I_0 + R_0
    dt = 0.01
    num_steps = int(t_max / dt)
    t = np.linspace(0, t_max, num_steps)
    S = np.zeros(num_steps)
    E = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    S[0] = S_0
    E[0] = E_0
    I[0] = I_0
    R[0] = R_0
    for i in range(1, num_steps):
        dS = -beta * S[i-1] * I[i-1] / N
        dE = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dI = sigma * E[i-1] - gamma * I[i-1]
        dR = gamma * I[i-1]
        S[i] = S[i-1] + dt * dS
        E[i] = E[i-1] + dt * dE
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
    return t, S, E, I, R

beta = 0.8
gamma = 0.2
sigma = 0.1
initial_conditions = (990, 10, 0, 0)
t_max = 100

t, S, E, I, R = seir_model(beta, gamma, sigma, initial_conditions, t_max)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()

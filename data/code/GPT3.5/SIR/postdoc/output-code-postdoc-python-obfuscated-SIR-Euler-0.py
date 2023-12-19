import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, init_infected, population, t_max, dt):
    n_steps = int(t_max / dt)
    S = np.zeros(n_steps + 1)
    I = np.zeros(n_steps + 1)
    R = np.zeros(n_steps + 1)
    t = np.linspace(0, t_max, n_steps + 1)
    
    S[0] = population - init_infected
    I[0] = init_infected
    R[0] = 0
    
    for i in range(n_steps):
        S[i+1] = S[i] - beta * S[i] * I[i] * dt / population
        I[i+1] = I[i] + (beta * S[i] * I[i] - gamma * I[i]) * dt
        R[i+1] = R[i] + gamma * I[i] * dt
    
    return S, I, R

beta = 0.5
gamma = 0.2
init_infected = 5
population = 1000
t_max = 100
dt = 0.1

S, I, R = SIR_model(beta, gamma, init_infected, population, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()

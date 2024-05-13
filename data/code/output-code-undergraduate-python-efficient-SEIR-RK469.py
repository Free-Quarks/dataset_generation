import numpy as np
import matplotlib.pyplot as plt


def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def run_seir_model(beta, sigma, gamma, S0, E0, I0, R0, t_max):
    y0 = S0, E0, I0, R0
    t = np.linspace(0, t_max, t_max+1)
    result = odeint(seir_model, y0, t, args=(beta, sigma, gamma))
    S, E, I, R = result.T
    return S, E, I, R


# Example usage
beta = 0.3
sigma = 0.1
gamma = 0.05
S0 = 0.99
E0 = 0.01
I0 = 0
R0 = 0
t_max = 100

S, E, I, R = run_seir_model(beta, sigma, gamma, S0, E0, I0, R0, t_max)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('SEIR Model')
plt.legend()
plt.show()

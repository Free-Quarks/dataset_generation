import numpy as np
import matplotlib.pyplot as plt

def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def run_seir_model(S0, E0, I0, R0, beta, sigma, gamma, t_max, n_steps):
    y0 = S0, E0, I0, R0
    t = np.linspace(0, t_max, n_steps)
    result = odeint(seir_model, y0, t, args=(beta, sigma, gamma))
    S, E, I, R = result.T
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()


S0 = 900
E0 = 100
I0 = 10
R0 = 0
beta = 0.5
sigma = 0.25
gamma = 0.1
t_max = 100
t_steps = 1000

run_seir_model(S0, E0, I0, R0, beta, sigma, gamma, t_max, t_steps)

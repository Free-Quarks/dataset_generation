import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sidarthe(y, t, beta, sigma, delta, kappa, alpha, rho, epsilon):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dydt = [-beta*S*I/N, beta*S*I/N - sigma*I - delta*I - kappa*I - alpha*I - rho*I, delta*I, alpha*I, sigma*I + kappa*I, rho*I, epsilon*I, epsilon*I]
    return dydt

def plot_results(t, S, I, D, A, R, T, H, E):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Deceased')
    plt.plot(t, A, label='Asymptomatic')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, T, label='Tested')
    plt.plot(t, H, label='Hospitalized')
    plt.plot(t, E, label='Exposed')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIDARTHE Model')
    plt.legend()
    plt.show()

y0 = [1000000, 10, 0, 0, 0, 0, 0, 0]
t = np.linspace(0, 100, 1000)
beta = 0.2
sigma = 0.04
delta = 0.01
kappa = 0.01
alpha = 0.01
rho = 0.01
epsilon = 0.01
sol = odeint(sidarthe, y0, t, args=(beta, sigma, delta, kappa, alpha, rho, epsilon))
S, I, D, A, R, T, H, E = sol.T
plot_results(t, S, I, D, A, R, T, H, E)

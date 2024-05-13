import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(N, T, beta, gamma, mu, sigma, rho, delta):
    # Total population, N.
    # Initial number of infected, recovered and deceased individuals, I0, R0 and D0.
    # Everyone else, S0, is susceptible to infection initially.
    S0, I0, D0, A0, R0, T0, H0, E0 = N-1, 1, 0, 0, 0, 0, 0, 0
    
    # Contact rate, beta, and mean recovery rate, gamma, and death rate, mu, and incubation rate, sigma, and hospitalization rate, rho, and mortality rate among hospitalized individuals, delta.
    
    # A grid of time points (in days)
    t = np.linspace(0, T, T)
    
    # Initial conditions vector
    y0 = S0, I0, D0, A0, R0, T0, H0, E0
    
    # Integrate the SIDARTHE equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, mu, sigma, rho, delta))
    S, I, D, A, R, T, H, E = ret.T
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(t, S, label='Susceptible')
    ax.plot(t, I, label='Infected')
    ax.plot(t, D, label='Deceased')
    ax.plot(t, A, label='Asymptomatic')
    ax.plot(t, R, label='Recovered')
    ax.plot(t, T, label='Tested')
    ax.plot(t, H, label='Hospitalized')
    ax.plot(t, E, label='Exposed')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('SIDARTHE Model')
    ax.legend()
    plt.show()


N = 1000000 # Total population
T = 180 # Simulation period (days)
beta = 0.2 # Contact rate
sigma = 1/5.2 # mean incubation period
rho = 0.2 # hospitalization rate
mu = 0.013 # mortality rate
gamma = 0.03 # recovery rate
R0 = beta/gamma # basic reproduction number

sidarthe_model(N, T, beta, gamma, mu, sigma, rho, delta)

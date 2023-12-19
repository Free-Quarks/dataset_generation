import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, population, initial_infected, total_days):
    # Set initial values
    initial_susceptible = population - initial_infected
    s = [initial_susceptible]
    i = [initial_infected]
    r = [0]
    t = [0]
    dt = 0.1
    N = int(total_days / dt)
    
    # Runge-Kutta method
    for _ in range(N):
        t.append(t[-1] + dt)
        s.append(s[-1] - dt * beta * s[-1] * i[-1] / population)
        i.append(i[-1] + dt * (beta * s[-2] * i[-1] / population - gamma * i[-1]))
        r.append(r[-1] + dt * gamma * i[-2])
    
    # Plotting
    plt.plot(t, s, label='Susceptible')
    plt.plot(t, i, label='Infected')
    plt.plot(t, r, label='Recovered')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


sir_model(0.3, 0.1, 1000, 1, 100)

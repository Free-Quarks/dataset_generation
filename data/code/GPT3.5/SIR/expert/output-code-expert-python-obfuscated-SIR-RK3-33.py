import numpy as np
import matplotlib.pyplot as plt

def sir_rk3(beta, gamma, population, initial_infected, days):
    def derivs(y, t):
        s, i, r = y
        ds_dt = -beta*s*i/population
        di_dt = beta*s*i/population - gamma*i
        dr_dt = gamma*i
        return [ds_dt, di_dt, dr_dt]

    dt = 0.1
    num_steps = int(days/dt)
    t = np.linspace(0, days, num_steps)
    y = np.zeros((num_steps, 3))
    y[0] = [population - initial_infected, initial_infected, 0]

    for i in range(num_steps-1):
        k1 = derivs(y[i], t[i])
        k2 = derivs(y[i]+0.5*dt*k1, t[i]+0.5*dt)
        k3 = derivs(y[i]+0.75*dt*k2, t[i]+0.75*dt)
        y[i+1] = y[i] + (2/9*k1 + 1/3*k2 + 4/9*k3)*dt

    plt.plot(t, y[:, 0], label='Susceptible')
    plt.plot(t, y[:, 1], label='Infected')
    plt.plot(t, y[:, 2], label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of Individuals')
    plt.legend()
    plt.show()
}

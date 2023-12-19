import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, population, initial_infected, days):
    # Initialize arrays
    num_steps = int(days) + 1
    dt = 1
    t = np.linspace(0, days, num_steps)
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    S[0] = population - initial_infected
    I[0] = initial_infected
    R[0] = 0

    # Runge-Kutta 2nd order method
    for i in range(1, num_steps):
        k1 = dt * (-beta * S[i-1] * I[i-1] / population)
        l1 = dt * (beta * S[i-1] * I[i-1] / population - gamma * I[i-1])
        k2 = dt * (-beta * (S[i-1] + k1/2) * (I[i-1] + l1/2) / population)
        l2 = dt * (beta * (S[i-1] + k1/2) * (I[i-1] + l1/2) / population - gamma * (I[i-1] + l1/2))
        S[i] = S[i-1] + k2
        I[i] = I[i-1] + l2
        R[i] = R[i-1] + gamma * (I[i-1] + l1/2)

    # Plotting
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()
}

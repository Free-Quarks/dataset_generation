import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(N, beta, gamma, I0, R0, t_max, h):
    # Define the differential equations for the SIR model
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Create a time array from 0 to t_max with step size h
    t = np.arange(0, t_max, h)

    # Set the initial conditions
    y0 = N - I0, I0, R0

    # Use the Runge-Kutta 4th order method to solve the differential equations
    ys = np.zeros((len(t), 3))
    ys[0] = y0
    for i in range(1, len(t)):
        k1 = h * deriv(ys[i-1], t[i-1], N, beta, gamma)
        k2 = h * deriv(ys[i-1] + 0.5 * k1, t[i-1] + 0.5 * h, N, beta, gamma)
        k3 = h * deriv(ys[i-1] + 0.5 * k2, t[i-1] + 0.5 * h, N, beta, gamma)
        k4 = h * deriv(ys[i-1] + k3, t[i-1] + h, N, beta, gamma)
        ys[i] = ys[i-1] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    # Extract the S, I, R values
    S, I, R = ys[:, 0], ys[:, 1], ys[:, 2]

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Example usage

# Parameters
N = 1000  # Total population
beta = 0.2  # Infection rate
gamma = 0.1  # Recovery rate
I0 = 1  # Initial number of infected individuals
R0 = 0  # Initial number of recovered individuals

# Simulation duration and step size
t_max = 100
h = 0.1

SIR_RK4(N, beta, gamma, I0, R0, t_max, h)

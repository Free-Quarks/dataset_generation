import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, N, I0, R0, T):
    # Function to implement the SIR model using RK2
    def derivs(y, t):
        S, I, R = y
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    # Initial conditions
    S0 = N - I0 - R0
    y0 = [S0, I0, R0]

    # Time points
    t = np.linspace(0, T, T+1)

    # Solve the ODE using RK2
    sol = odeint(derivs, y0, t)
    S, I, R = sol.T

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 10
R0 = 0
T = 100

SIR_RK2(beta, gamma, N, I0, R0, T)

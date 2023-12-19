import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(S0, I0, R0, beta, gamma, tmax, dt):
    '''
    Simulate SIR model using 3rd order Runge-Kutta method.

    Parameters:
    ----------
    S0, I0, R0 : int
        Initial number of susceptible, infected, and recovered individuals
    beta : float
        The infection rate
    gamma : float
        The recovery rate
    tmax : int
        Maximum time to simulate
    dt : float
        Time step size

    Returns:
    -------
    S, I, R : ndarray
        Arrays containing the number of susceptible, infected, and recovered individuals at each time step
    '''
    N = S0 + I0 + R0
    t = np.arange(0, tmax, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1 = dt * (-beta * S[i-1] * I[i-1] / N)
        k2 = dt * (-beta * (S[i-1] + 0.5 * k1) * (I[i-1] + 0.5 * k1) / N)
        k3 = dt * (-beta * (S[i-1] - k1 + 2 * k2) * (I[i-1] - k1 + 2 * k2) / N)
        S[i] = S[i-1] + k1
        I[i] = I[i-1] + k1
        R[i] = R[i-1] + gamma * I[i-1]

    return S, I, R


# Example usage
S0 = 999
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
tmax = 100
dt = 0.1

S, I, R = SIR_RK3(S0, I0, R0, beta, gamma, tmax, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt


# Define the system of differential equations

def serid(y, t, beta, gamma, delta):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - delta * E
    dIdt = delta * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


# Runge-Kutta 4th order method

def rk4(y0, t, h, beta, gamma, delta):
    N = len(t)
    S = np.zeros(N)
    E = np.zeros(N)
    I = np.zeros(N)
    R = np.zeros(N)
    S[0], E[0], I[0], R[0] = y0
    for i in range(1, N):
        k1 = h * np.array(serid([S[i-1], E[i-1], I[i-1], R[i-1]], t[i-1], beta, gamma, delta))
        k2 = h * np.array(serid([S[i-1] + k1[0]/2, E[i-1] + k1[1]/2, I[i-1] + k1[2]/2, R[i-1] + k1[3]/2], t[i-1] + h/2, beta, gamma, delta))
        k3 = h * np.array(serid([S[i-1] + k2[0]/2, E[i-1] + k2[1]/2, I[i-1] + k2[2]/2, R[i-1] + k2[3]/2], t[i-1] + h/2, beta, gamma, delta))
        k4 = h * np.array(serid([S[i-1] + k3[0], E[i-1] + k3[1], I[i-1] + k3[2], R[i-1] + k3[3]], t[i-1] + h, beta, gamma, delta))
        S[i] = S[i-1] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        E[i] = E[i-1] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        I[i] = I[i-1] + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        R[i] = R[i-1] + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6
    return S, E, I, R


def main():
    # Define the parameters
    beta = 0.2
    gamma = 0.1
    delta = 0.05
    t = np.linspace(0, 100, 1000)
    h = t[1] - t[0]
    y0 = [0.99, 0.01, 0, 0]

    # Solve the system using RK4
    S, E, I, R = rk4(y0, t, h, beta, gamma, delta)

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Fraction of Population')
    plt.title('SERID Model')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt


def seir_model(N, beta, gamma, sigma, I0, E0, R0, T):
    S0 = N - I0 - R0 - E0
    S = [S0]
    I = [I0]
    E = [E0]
    R = [R0]
    dt = 1  # time step
    t = np.arange(0, T, dt)

    for i in range(1, len(t)):
        dS = -beta * S[i - 1] * I[i - 1] / N
        dE = beta * S[i - 1] * I[i - 1] / N - sigma * E[i - 1]
        dI = sigma * E[i - 1] - gamma * I[i - 1]
        dR = gamma * I[i - 1]
        S.append(S[i - 1] + dt * dS)
        E.append(E[i - 1] + dt * dE)
        I.append(I[i - 1] + dt * dI)
        R.append(R[i - 1] + dt * dR)

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SEIR Model')
    plt.show()


seir_model(100000, 0.3, 0.1, 0.2, 100, 10, 0, 200)

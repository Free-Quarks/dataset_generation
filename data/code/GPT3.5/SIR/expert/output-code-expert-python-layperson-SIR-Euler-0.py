import numpy as np
import matplotlib.pyplot as plt


def simulate_SIR(N, I0, R0, beta, gamma, days):
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    dt = 1
    t = np.arange(0, days, dt)

    for _ in t[1:]:
        dS = -beta * S[-1] * I[-1] / N
        dI = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dR = gamma * I[-1]

        S.append(S[-1] + dS * dt)
        I.append(I[-1] + dI * dt)
        R.append(R[-1] + dR * dt)

    return S, I, R


def plot_SIR(S, I, R):
    plt.plot(S, label='Susceptible')
    plt.plot(I, label='Infected')
    plt.plot(R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


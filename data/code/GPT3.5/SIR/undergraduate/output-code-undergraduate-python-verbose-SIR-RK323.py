import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, N, I0, T):
    def f(t, y):
        S, I, R = y
        return [-beta*S*I/N, beta*S*I/N-gamma*I, gamma*I]

    t = np.linspace(0, T, int(T/0.1)+1)
    y0 = [N-I0, I0, 0]
    h = t[1] - t[0]

    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    S[0], I[0], R[0] = y0

    for i in range(len(t)-1):
        k1 = f(t[i], [S[i], I[i], R[i]])
        k2 = f(t[i] + h/2, [S[i] + h/2 * k1[0], I[i] + h/2 * k1[1], R[i] + h/2 * k1[2]])
        k3 = f(t[i] + h, [S[i] - h * k1[0] + 2 * h * k2[0], I[i] - h * k1[1] + 2 * h * k2[1], R[i] - h * k1[2] + 2 * h * k2[2]])

        S[i+1] = S[i] + h/6 * (k1[0] + 4 * k2[0] + k3[0])
        I[i+1] = I[i] + h/6 * (k1[1] + 4 * k2[1] + k3[1])
        R[i+1] = R[i] + h/6 * (k1[2] + 4 * k2[2] + k3[2])

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model using RK3')
    plt.legend()
    plt.show()


SIR_RK3(beta=0.2, gamma=0.1, N=1000, I0=1, T=100)

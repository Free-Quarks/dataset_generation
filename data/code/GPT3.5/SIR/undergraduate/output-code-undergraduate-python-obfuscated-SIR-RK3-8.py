import numpy as np
import matplotlib.pyplot as plt

def sir_rk3(beta, gamma, N, I0, days):
    def f(t, y):
        S, I, R = y
        return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]

    y0 = [N-I0, I0, 0]
    t = np.linspace(0, days, days+1)
    h = t[1]-t[0]

    S = np.zeros(days+1)
    I = np.zeros(days+1)
    R = np.zeros(days+1)

    S[0], I[0], R[0] = y0

    for i in range(1, days+1):
        k1 = h*f(t[i-1], [S[i-1], I[i-1], R[i-1]])
        k2 = h*f(t[i-1]+h/2, [S[i-1]+k1[0]/2, I[i-1]+k1[1]/2, R[i-1]+k1[2]/2])
        k3 = h*f(t[i-1]+h, [S[i-1]-k1[0]+2*k2[0], I[i-1]-k1[1]+2*k2[1], R[i-1]-k1[2]+2*k2[2]])

        S[i] = S[i-1] + (k1[0] + 4*k2[0] + k3[0])/6
        I[i] = I[i-1] + (k1[1] + 4*k2[1] + k3[1])/6
        R[i] = R[i-1] + (k1[2] + 4*k2[2] + k3[2])/6

    plt.plot(t, S, label='S')
    plt.plot(t, I, label='I')
    plt.plot(t, R, label='R')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SIR Model Simulation')
    plt.show()


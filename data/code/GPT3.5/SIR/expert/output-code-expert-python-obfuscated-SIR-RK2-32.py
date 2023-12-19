import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, N, I0, R0, t_end, dt):
    # Initial conditions
    S0 = N - I0 - R0
    Y0 = np.array([S0, I0, R0])
    Y = np.zeros((int(t_end/dt)+1, 3))
    Y[0] = Y0

    # RK2 integration
    for i in range(1, int(t_end/dt)+1):
        S_prev, I_prev, R_prev = Y[i-1]
        k1 = dt * np.array([-beta * S_prev * I_prev / N, beta * S_prev * I_prev / N - gamma * I_prev, gamma * I_prev])
        k2 = dt * np.array([-beta * (S_prev + 0.5 * k1[0]) * (I_prev + 0.5 * k1[1]) / N,
                            beta * (S_prev + 0.5 * k1[0]) * (I_prev + 0.5 * k1[1]) / N - gamma * (I_prev + 0.5 * k1[1]),
                            gamma * (I_prev + 0.5 * k1[1])])
        Y[i] = Y[i-1] + k2

    return Y[:,0], Y[:,1], Y[:,2]


# Example usage
beta = 0.3
gamma = 0.1
N = 1000
I0 = 10
R0 = 0
t_end = 100
dt = 0.1
S, I, R = SIR_RK2(beta, gamma, N, I0, R0, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK2 Integration')
plt.legend()
plt.show()


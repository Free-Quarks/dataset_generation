import numpy as np
import matplotlib.pyplot as plt


def SIDARTHE_RK3(beta, gamma, delta, alpha, rho, theta, epsilon, N, I0, D0, A0, R0, T, dt):
    t = np.linspace(0, T, int(T/dt) + 1)
    S = np.zeros(t.shape)
    I = np.zeros(t.shape)
    D = np.zeros(t.shape)
    A = np.zeros(t.shape)
    R = np.zeros(t.shape)
    E = np.zeros(t.shape)
    T = np.zeros(t.shape)
    H = np.zeros(t.shape)

    S[0] = N
    I[0] = I0
    D[0] = D0
    A[0] = A0
    R[0] = R0
    E[0] = 0
    T[0] = 0
    H[0] = 0

    for i in range(len(t)-1):
        dSdt = - beta * S[i] * (I[i] + alpha * A[i]) / N
        dEdt = beta * S[i] * (I[i] + alpha * A[i]) / N - delta * E[i]
        dIdt = delta * (1 - rho) * E[i] - gamma * I[i]
        dDdt = delta * rho * E[i]
        dAdt = delta * theta * rho * E[i] - epsilon * A[i]
        dRdt = gamma * I[i] + epsilon * A[i]
        dHdt = delta * (1 - rho) * E[i]

        S[i+1] = S[i] + dt * dSdt
        E[i+1] = E[i] + dt * dEdt
        I[i+1] = I[i] + dt * dIdt
        D[i+1] = D[i] + dt * dDdt
        A[i+1] = A[i] + dt * dAdt
        R[i+1] = R[i] + dt * dRdt
        H[i+1] = H[i] + dt * dHdt
        T[i+1] = S[i+1] + E[i+1] + I[i+1] + D[i+1] + A[i+1] + R[i+1] + H[i+1]

    return t, S, E, I, D, A, R, H, T


beta = 0.5
gamma = 0.1
alpha = 0.2
rho = 0.8
theta = 0.2
epsilon = 0.05
N = 10000
I0 = 10
D0 = 0
A0 = 0
R0 = 0
T = 100
dt = 0.1

t, S, E, I, D, A, R, H, T = SIDARTHE_RK3(beta, gamma, alpha, rho, theta, epsilon, N, I0, D0, A0, R0, T, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Dead')
plt.plot(t, A, label='Asymptomatic')
plt.plot(t, R, label='Recovered')
plt.plot(t, H, label='Hospitalized')
plt.plot(t, T, label='Total')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

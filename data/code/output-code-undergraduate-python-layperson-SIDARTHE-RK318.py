import numpy as np
import matplotlib.pyplot as plt


def sidarthe(y0, t, beta, gamma, delta, alpha, rho):
    S, I, D, A, R, T, H, E = y0
    N = sum(y0)
    dt = t[1] - t[0]
    n = len(t)
    S_res = np.zeros(n)
    I_res = np.zeros(n)
    D_res = np.zeros(n)
    A_res = np.zeros(n)
    R_res = np.zeros(n)
    T_res = np.zeros(n)
    H_res = np.zeros(n)
    E_res = np.zeros(n)
    S_res[0] = S
    I_res[0] = I
    D_res[0] = D
    A_res[0] = A
    R_res[0] = R
    T_res[0] = T
    H_res[0] = H
    E_res[0] = E
    for i in range(1, n):
        S_to_E = (beta * S * (I + alpha * A)) / N
        S_to_A = rho * S_to_E
        S_to_I = S_to_E - S_to_A
        A_to_R = delta * A
        I_to_H = gamma * I
        I_to_R = (1 - gamma) * I
        I_to_T = alpha * gamma * I
        H_to_R = (1 - rho) * H
        H_to_T = rho * H
        S -= S_to_E * dt
        E += (S_to_E - S_to_A) * dt
        A += (S_to_A - A_to_R) * dt
        I += (S_to_I - I_to_H - I_to_R - I_to_T) * dt
        R += (I_to_R + H_to_R) * dt
        T += (I_to_T + H_to_T) * dt
        H += I_to_H * dt
        S_res[i] = S
        E_res[i] = E
        A_res[i] = A
        I_res[i] = I
        R_res[i] = R
        T_res[i] = T
        H_res[i] = H
    return S_res, E_res, A_res, I_res, R_res, T_res, H_res


y0 = [9999, 1, 0, 0, 0, 0, 0, 0]
t = np.linspace(0, 100, 1000)
beta = 0.2
gamma = 0.1
delta = 0.1
alpha = 0.2ho = 0.2

S_res, E_res, A_res, I_res, R_res, T_res, H_res = sidarthe(y0, t, beta, gamma, delta, alpha, rho)

plt.plot(t, S_res, label='Susceptible')
plt.plot(t, E_res, label='Exposed')
plt.plot(t, A_res, label='Asymptomatic')
plt.plot(t, I_res, label='Infected')
plt.plot(t, R_res, label='Recovered')
plt.plot(t, T_res, label='Transferred')
plt.plot(t, H_res, label='Hospitalized')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.legend()
plt.title('SIDARTHE Model')
plt.show()

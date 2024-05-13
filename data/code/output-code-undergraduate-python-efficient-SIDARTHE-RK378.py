import numpy as np
import matplotlib.pyplot as plt

def sidarthe_rk3(N, T, dt, S_0, I_0, D_0, A_0, R_0, T_0, H_0, E_0, beta, mu, sigma_1, sigma_2, sigma_3, theta, phi, delta_1, delta_2, delta_3, gamma, omega):
    num_steps = int(T / dt)
    t = np.linspace(0, T, num_steps + 1)
    S = np.zeros(num_steps + 1)
    I = np.zeros(num_steps + 1)
    D = np.zeros(num_steps + 1)
    A = np.zeros(num_steps + 1)
    R = np.zeros(num_steps + 1)
    T = np.zeros(num_steps + 1)
    H = np.zeros(num_steps + 1)
    E = np.zeros(num_steps + 1)
    S[0] = S_0
    I[0] = I_0
    D[0] = D_0
    A[0] = A_0
    R[0] = R_0
    T[0] = T_0
    H[0] = H_0
    E[0] = E_0
    for i in range(num_steps):
        # Define the differential equations
        dS = -beta * S[i] * (I[i] + theta * A[i]) / N
        dI = beta * S[i] * (I[i] + theta * A[i]) / N - (sigma_1 + phi * sigma_3) * I[i] - delta_1 * I[i]
        dD = delta_1 * I[i] + delta_2 * T[i] + delta_3 * H[i] - (mu + gamma) * D[i]
        dA = sigma_1 * I[i] + sigma_2 * T[i] - (sigma_3 + delta_1) * A[i]
        dR = phi * sigma_3 * I[i] + gamma * D[i]
        dT = sigma_2 * A[i] - (sigma_1 + delta_2) * T[i]
        dH = delta_3 * D[i] - (delta_1 + delta_2) * H[i]
        dE = sigma_3 * (I[i] + theta * A[i]) - (sigma_1 + sigma_2) * E[i]
        # Update the variables using RK3 method
        k1_S = dt * dS
        k1_I = dt * dI
        k1_D = dt * dD
        k1_A = dt * dA
        k1_R = dt * dR
        k1_T = dt * dT
        k1_H = dt * dH
        k1_E = dt * dE
        k2_S = dt * (-beta * (S[i] + k1_S / 2) * (I[i] + k1_I / 2 + theta * (A[i] + k1_A / 2)) / N)
        k2_I = dt * (beta * (S[i] + k1_S / 2) * (I[i] + k1_I / 2 + theta * (A[i] + k1_A / 2)) / N - (sigma_1 + phi * sigma_3) * (I[i] + k1_I / 2) - delta_1 * (I[i] + k1_I / 2))
        k2_D = dt * (delta_1 * (I[i] + k1_I / 2) + delta_2 * (T[i] + k1_T / 2) + delta_3 * (H[i] + k1_H / 2)) - (mu + gamma) * (D[i] + k1_D / 2)
        k2_A = dt * (sigma_1 * (I[i] + k1_I / 2) + sigma_2 * (T[i] + k1_T / 2)) - (sigma_3 + delta_1) * (A[i] + k1_A / 2)
        k2_R = dt * (phi * sigma_3 * (I[i] + k1_I / 2) + gamma * (D[i] + k1_D / 2))
        k2_T = dt * (sigma_2 * (A[i] + k1_A / 2)) - (sigma_1 + delta_2) * (T[i] + k1_T / 2)
        k2_H = dt * (delta_3 * (D[i] + k1_D / 2)) - (delta_1 + delta_2) * (H[i] + k1_H / 2)
        k2_E = dt * (sigma_3 * (I[i] + k1_I / 2 + theta * (A[i] + k1_A / 2))) - (sigma_1 + sigma_2) * (E[i] + k1_E / 2)
        k3_S = dt * (-beta * (S[i] - k1_S + 2 * k2_S) * (I[i] - k1_I + 2 * k2_I + theta * (A[i] - k1_A + 2 * k2_A)) / N)
        k3_I = dt * (beta * (S[i] - k1_S + 2 * k2_S) * (I[i] - k1_I + 2 * k2_I + theta * (A[i] - k1_A + 2 * k2_A)) / N - (sigma_1 + phi * sigma_3) * (I[i] - k1_I + 2 * k2_I) - delta_1 * (I[i] - k1_I + 2 * k2_I))
        k3_D = dt * (delta_1 * (I[i] - k1_I + 2 * k2_I) + delta_2 * (T[i] - k1_T + 2 * k2_T) + delta_3 * (H[i] - k1_H + 2 * k2_H)) - (mu + gamma) * (D[i] - k1_D + 2 * k2_D)
        k3_A = dt * (sigma_1 * (I[i] - k1_I + 2 * k2_I) + sigma_2 * (T[i] - k1_T + 2 * k2_T)) - (sigma_3 + delta_1) * (A[i] - k1_A + 2 * k2_A)
        k3_R = dt * (phi * sigma_3 * (I[i] - k1_I + 2 * k2_I) + gamma * (D[i] - k1_D + 2 * k2_D))
        k3_T = dt * (sigma_2 * (A[i] - k1_A + 2 * k2_A)) - (sigma_1 + delta_2) * (T[i] - k1_T + 2 * k2_T)
        k3_H = dt * (delta_3 * (D[i] - k1_D + 2 * k2_D)) - (delta_1 + delta_2) * (H[i] - k1_H + 2 * k2_H)
        k3_E = dt * (sigma_3 * (I[i] - k1_I + 2 * k2_I + theta * (A[i] - k1_A + 2 * k2_A))) - (sigma_1 + sigma_2) * (E[i] - k1_E + 2 * k2_E)
        S[i+1] = S[i] + (k1_S + 4 * k2_S + k3_S) / 6
        I[i+1] = I[i] + (k1_I + 4 * k2_I + k3_I) / 6
        D[i+1] = D[i] + (k1_D + 4 * k2_D + k3_D) / 6
        A[i+1] = A[i] + (k1_A + 4 * k2_A + k3_A) / 6
        R[i+1] = R[i] + (k1_R + 4 * k2_R + k3_R) / 6
        T[i+1] = T[i] + (k1_T + 4 * k2_T + k3_T) / 6
        H[i+1] = H[i] + (k1_H + 4 * k2_H + k3_H) / 6
        E[i+1] = E[i] + (k1_E + 4 * k2_E + k3_E) / 6
    return S, I, D, A, R, T, H, E


N = 1000000
T = 200
dt = 0.1

S_0 = N - 1
I_0 = 1
D_0 = 0
A_0 = 0
R_0 = 0
T_0 = 0
H_0 = 0
E_0 = 0

beta = 0.5
mu = 0.01
sigma_1 = 0.1
sigma_2 = 0.2
sigma_3 = 0.05
theta = 0.3
phi = 0.2
delta_1 = 0.1
delta_2 = 0.2
delta_3 = 0.15
gamma = 0.1
omega = 0.01

S, I, D, A, R, T, H, E = sidarthe_rk3(N, T, dt, S_0, I_0, D_0, A_0, R_0, T_0, H_0, E_0, beta, mu, sigma_1, sigma_2, sigma_3, theta, phi, delta_1, delta_2, delta_3, gamma, omega)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(D, label='Deceased')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Tested')
plt.plot(H, label='Hospitalized')
plt.plot(E, label='Exposed')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(S0, I0, R0, beta, gamma, t_max, dt):
    N = S0 + I0 + R0
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for t in range(t_max-1):
        dsdt = -beta * S[t] * I[t] / N
        didt = beta * S[t] * I[t] / N - gamma * I[t]
        drdt = gamma * I[t]

        k1_s = dt * dsdt
        k1_i = dt * didt
        k1_r = dt * drdt

        k2_s = dt * (-beta * (S[t] + 0.5 * k1_s) * (I[t] + 0.5 * k1_i) / N)
        k2_i = dt * (beta * (S[t] + 0.5 * k1_s) * (I[t] + 0.5 * k1_i) / N - gamma * (I[t] + 0.5 * k1_i))
        k2_r = dt * (gamma * (I[t] + 0.5 * k1_i))

        k3_s = dt * (-beta * (S[t] + 0.5 * k2_s) * (I[t] + 0.5 * k2_i) / N)
        k3_i = dt * (beta * (S[t] + 0.5 * k2_s) * (I[t] + 0.5 * k2_i) / N - gamma * (I[t] + 0.5 * k2_i))
        k3_r = dt * (gamma * (I[t] + 0.5 * k2_i))

        k4_s = dt * (-beta * (S[t] + k3_s) * (I[t] + k3_i) / N)
        k4_i = dt * (beta * (S[t] + k3_s) * (I[t] + k3_i) / N - gamma * (I[t] + k3_i))
        k4_r = dt * (gamma * (I[t] + k3_i))

        S[t+1] = S[t] + (1/6) * (k1_s + 2*k2_s + 2*k3_s + k4_s)
        I[t+1] = I[t] + (1/6) * (k1_i + 2*k2_i + 2*k3_i + k4_i)
        R[t+1] = R[t] + (1/6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)

    return S, I, R


def plot_SIR(S, I, R):
    t = np.arange(len(S))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SIR Model')
    plt.show()


# Example usage
S0 = 999
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
t_max = 100
dt = 0.1

S, I, R = SIR_RK4(S0, I0, R0, beta, gamma, t_max, dt)
plot_SIR(S, I, R)

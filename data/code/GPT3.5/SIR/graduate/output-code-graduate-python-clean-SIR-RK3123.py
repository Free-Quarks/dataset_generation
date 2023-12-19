import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, N, I0, T):
    def dSdt(S, I):
        return -beta * S * I / N

    def dIdt(S, I):
        return beta * S * I / N - gamma * I

    def dRdt(I):
        return gamma * I

    def RK3_step(S, I, R, dt):
        k1_S = dSdt(S, I)
        k1_I = dIdt(S, I)
        k1_R = dRdt(I)

        k2_S = dSdt(S + dt / 2 * k1_S, I + dt / 2 * k1_I)
        k2_I = dIdt(S + dt / 2 * k1_S, I + dt / 2 * k1_I)
        k2_R = dRdt(I + dt / 2 * k1_I)

        k3_S = dSdt(S + dt * (-k1_S + 2 * k2_S), I + dt * (-k1_I + 2 * k2_I))
        k3_I = dIdt(S + dt * (-k1_S + 2 * k2_S), I + dt * (-k1_I + 2 * k2_I))
        k3_R = dRdt(I + dt * (-k1_I + 2 * k2_I))

        S_new = S + dt / 6 * (k1_S + 4 * k2_S + k3_S)
        I_new = I + dt / 6 * (k1_I + 4 * k2_I + k3_I)
        R_new = R + dt / 6 * (k1_R + 4 * k2_R + k3_R)

        return S_new, I_new, R_new

    S = N - I0
    I = I0
    R = 0
    dt = 0.01
    timesteps = int(T / dt)

    S_values = [S]
    I_values = [I]
    R_values = [R]

    for _ in range(timesteps):
        S, I, R = RK3_step(S, I, R, dt)
        S_values.append(S)
        I_values.append(I)
        R_values.append(R)

    return S_values, I_values, R_values


# Example usage
beta = 0.3
gamma = 0.1
N = 1000
I0 = 1
T = 100

S_values, I_values, R_values = SIR_RK3(beta, gamma, N, I0, T)

plt.plot(S_values, label='Susceptible')
plt.plot(I_values, label='Infected')
plt.plot(R_values, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()

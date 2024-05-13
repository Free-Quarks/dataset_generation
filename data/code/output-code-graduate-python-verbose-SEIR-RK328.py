import numpy as np
import matplotlib.pyplot as plt

def SEIR_RK3(beta, sigma, gamma, N, I0, R0, t_max, dt):
    def derivs(state, t):
        S, E, I, R = state
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    E = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    S[0] = N - I0 - R0
    E[0] = 0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        state = S[i - 1], E[i - 1], I[i - 1], R[i - 1]
        k1 = dt * np.array(derivs(state, t[i - 1]))
        k2 = dt * np.array(derivs(state + 0.5 * k1, t[i - 1] + 0.5 * dt))
        k3 = dt * np.array(derivs(state - k1 + 2 * k2, t[i - 1] + dt))
        state += (k1 + 4 * k2 + k3) / 6
        S[i], E[i], I[i], R[i] = state

    return S, E, I, R

# Example usage:
N = 1000
I0 = 1
R0 = 0
beta = 0.2
sigma = 1/5
gamma = 1/10
t_max = 100
dt = 0.1

S, E, I, R = SEIR_RK3(beta, sigma, gamma, N, I0, R0, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIR Model using RK3')
plt.legend()
plt.show()

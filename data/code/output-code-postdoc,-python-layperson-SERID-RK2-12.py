import numpy as np
import matplotlib.pyplot as plt

def serid_rk2(S0, E0, R0, I0, D0, beta, alpha, gamma, delta, t_max, h):
    def derivs(S, E, R, I, D, beta, alpha, gamma, delta):
        N = S + E + R + I + D
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - alpha * E
        dRdt = gamma * I
        dIdt = alpha * E - gamma * I - delta * I
        dDdt = delta * I
        return dSdt, dEdt, dRdt, dIdt, dDdt

    t = np.arange(0, t_max, h)
    S = np.zeros_like(t)
    E = np.zeros_like(t)
    R = np.zeros_like(t)
    I = np.zeros_like(t)
    D = np.zeros_like(t)
    S[0] = S0
    E[0] = E0
    R[0] = R0
    I[0] = I0
    D[0] = D0

    for i in range(1, len(t)):
        S_half = S[i-1] + h * derivs(S[i-1], E[i-1], R[i-1], I[i-1], D[i-1], beta, alpha, gamma, delta)[0] / 2
        E_half = E[i-1] + h * derivs(S[i-1], E[i-1], R[i-1], I[i-1], D[i-1], beta, alpha, gamma, delta)[1] / 2
        R_half = R[i-1] + h * derivs(S[i-1], E[i-1], R[i-1], I[i-1], D[i-1], beta, alpha, gamma, delta)[2] / 2
        I_half = I[i-1] + h * derivs(S[i-1], E[i-1], R[i-1], I[i-1], D[i-1], beta, alpha, gamma, delta)[3] / 2
        D_half = D[i-1] + h * derivs(S[i-1], E[i-1], R[i-1], I[i-1], D[i-1], beta, alpha, gamma, delta)[4] / 2

        S[i] = S[i-1] + h * derivs(S_half, E_half, R_half, I_half, D_half, beta, alpha, gamma, delta)[0]
        E[i] = E[i-1] + h * derivs(S_half, E_half, R_half, I_half, D_half, beta, alpha, gamma, delta)[1]
        R[i] = R[i-1] + h * derivs(S_half, E_half, R_half, I_half, D_half, beta, alpha, gamma, delta)[2]
        I[i] = I[i-1] + h * derivs(S_half, E_half, R_half, I_half, D_half, beta, alpha, gamma, delta)[3]
        D[i] = D[i-1] + h * derivs(S_half, E_half, R_half, I_half, D_half, beta, alpha, gamma, delta)[4]

    return t, S, E, R, I, D


# Example usage
S0 = 1000
E0 = 0
R0 = 0
I0 = 1
D0 = 0
beta = 0.2
alpha = 0.1
gamma = 0.1
delta = 0.01

t_max = 100
h = 0.1

t, S, E, R, I, D = serid_rk2(S0, E0, R0, I0, D0, beta, alpha, gamma, delta, t_max, h)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, R, label='Recovered')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SERID Model using RK2')
plt.legend()
plt.show()

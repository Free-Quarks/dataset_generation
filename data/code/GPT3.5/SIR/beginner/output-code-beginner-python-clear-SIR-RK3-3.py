import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, R0, T):
    def derivs(u, t):
        S, I, R = u
        dS_dt = -beta * S * I / N
        dI_dt = beta * S * I / N - gamma * I
        dR_dt = gamma * I
        return [dS_dt, dI_dt, dR_dt]

    S0 = N - I0 - R0
    u0 = [S0, I0, R0]
    t = np.linspace(0, T, int(T))
    dt = t[1] - t[0]

    u = np.zeros((len(t), len(u0)))
    u[0] = u0

    for i in range(1, len(t)):
        u_star = u[i-1] + dt * derivs(u[i-1], t[i-1])
        u[i] = 0.75 * u[i-1] + 0.25 * (u_star + dt * derivs(u_star, t[i]))

    return t, u[:, 0], u[:, 1], u[:, 2]


beta = 0.3
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 100

t, S, I, R = SIR_RK3(beta, gamma, N, I0, R0, T)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model - RK3')
plt.legend()
plt.show()

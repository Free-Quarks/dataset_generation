import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, N, I0, R0, t_max, dt):
    def SIR_model(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    t = np.arange(0, t_max, dt)
    y0 = N - I0, I0, R0
    
    result = np.zeros((len(t), 3))
    result[0] = y0
    
    for i in range(1, len(t)):
        k1 = SIR_model(result[i - 1], t[i - 1], beta, gamma)
        k2 = SIR_model(result[i - 1] + 0.5 * dt * k1, t[i - 1] + 0.5 * dt, beta, gamma)
        k3 = SIR_model(result[i - 1] + 0.5 * dt * k2, t[i - 1] + 0.5 * dt, beta, gamma)
        k4 = SIR_model(result[i - 1] + dt * k3, t[i], beta, gamma)
        result[i] = result[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return result[:, 0], result[:, 1], result[:, 2]

# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_max = 100
dt = 0.1
S, I, R = SIR_RK4(beta, gamma, N, I0, R0, t_max, dt)

plt.plot(S, label='S')
plt.plot(I, label='I')
plt.plot(R, label='R')
plt.legend()
plt.show()

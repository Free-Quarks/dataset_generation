import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(N, I0, R0, beta, gamma, t_max):
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, t_max, num=t_max+1)

    S0 = N - I0 - R0
    y0 = S0, I0, R0
    result = np.zeros((3, t_max+1))
    result[:, 0] = y0

    for i in range(t_max):
        h = t[i+1] - t[i]
        k1 = h * deriv(result[:, i], t[i], N, beta, gamma)
        k2 = h * deriv(result[:, i] + 0.5 * k1, t[i] + 0.5 * h, N, beta, gamma)
        k3 = h * deriv(result[:, i] + 0.5 * k2, t[i] + 0.5 * h, N, beta, gamma)
        k4 = h * deriv(result[:, i] + k3, t[i+1], N, beta, gamma)
        result[:, i+1] = result[:, i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    return result


N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_max = 100

result = SIR_RK4(N, I0, R0, beta, gamma, t_max)

plt.plot(result[0], label='Susceptible')
plt.plot(result[1], label='Infected')
plt.plot(result[2], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK4')
plt.legend()
plt.show()

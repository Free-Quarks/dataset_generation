import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, I0, N, t_max):
    def deriv(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, t_max, t_max+1)
    y0 = N - I0, I0, 0
    ret = odeint(deriv, y0, t, args=(beta, gamma))
    S, I, R = ret.T

    return S, I, R


beta = 0.2
gamma = 0.1
I0 = 1
N = 1000
t_max = 100

S, I, R = SIR_model(beta, gamma, I0, N, t_max)

plt.plot(S, 'b', label='Susceptible')
plt.plot(I, 'r', label='Infected')
plt.plot(R, 'g', label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()

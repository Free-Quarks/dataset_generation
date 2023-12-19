import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, y0, t):
    def deriv(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    y = y0
    sol = []
    for ti in t:
        sol.append(y)
        dydt = deriv(y, ti, beta, gamma)
        y = y + (1/6)*(dydt + 2*deriv(y + 0.5*dydt, ti + 0.5, beta, gamma) + 2*deriv(y + 0.5*dydt, ti + 0.5, beta, gamma) + deriv(y + dydt, ti + 1, beta, gamma))
    
    return np.array(sol)

beta = 0.2
gamma = 0.1
y0 = [0.99, 0.01, 0]
t = np.linspace(0, 100, 100)

sol = sir_model(beta, gamma, y0, t)

plt.plot(t, sol[:, 0], label='S')
plt.plot(t, sol[:, 1], label='I')
plt.plot(t, sol[:, 2], label='R')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('SIR Model')
plt.legend()
plt.grid(True)
plt.show()

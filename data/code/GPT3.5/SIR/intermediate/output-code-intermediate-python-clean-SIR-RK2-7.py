import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(N, beta, gamma, I0, t_end):
    
    def f(t, y):
        S, I, R = y
        return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]
    
    def RK2(t, y, h):
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        y_next = y + k2
        return y_next
    
    S0 = N - I0
    y0 = [S0, I0, 0]
    
    t = np.arange(0, t_end + 1)
    Y = np.zeros((t.size, 3))
    Y[0] = y0
    
    for i in range(t.size - 1):
        Y[i+1] = RK2(t[i], Y[i], 1)
    
    return t, Y[:, 0], Y[:, 1], Y[:, 2]

N = 1000
beta = 0.2
gamma = 0.1
I0 = 1

t, S, I, R = SIR_RK2(N, beta, gamma, I0, 100)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model using RK2')
plt.legend()
plt.grid(True)
plt.show()


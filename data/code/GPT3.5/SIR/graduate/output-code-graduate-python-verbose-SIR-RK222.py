```python
import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, N, I0, R0, T):
    def SIR_model(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t_values = np.linspace(0, T, T+1)
    h = t_values[1] - t_values[0]

    y_values = np.zeros((T+1, 3))
    y_values[0] = N - I0, I0, R0

    for i in range(T):
        t = t_values[i]
        y = y_values[i]

        k1 = h * SIR_model(t, y)
        k2 = h * SIR_model(t + h/2, y + k1/2)
        y_values[i+1] = y + k2

    return y_values[:, 0], y_values[:, 1], y_values[:, 2]


beta = 0.3
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 100

S, I, R = SIR_RK2(beta, gamma, N, I0, R0, T)

plt.plot(range(T+1), S, label='Susceptible')
plt.plot(range(T+1), I, label='Infected')
plt.plot(range(T+1), R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()
```

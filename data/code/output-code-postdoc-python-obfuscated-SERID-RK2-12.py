```python
import numpy as np
import matplotlib.pyplot as plt


# Function implementing the model

def serid_rk2(S0, E0, I0, R0, beta, gamma, delta, alpha, t_max, dt):
    N = S0 + E0 + I0 + R0
    S = np.zeros(t_max)
    E = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    for i in range(1, t_max):
        k1 = -beta * S[i-1] * I[i-1] / N
        l1 = beta * S[i-1] * I[i-1] / N - delta * E[i-1]
        m1 = delta * E[i-1] - gamma * I[i-1]
        n1 = gamma * I[i-1] - alpha * R[i-1]

        k2 = -beta * (S[i-1] + dt/2 * k1) * (I[i-1] + dt/2 * m1) / N
        l2 = beta * (S[i-1] + dt/2 * k1) * (I[i-1] + dt/2 * m1) / N - delta * (E[i-1] + dt/2 * l1)
        m2 = delta * (E[i-1] + dt/2 * l1) - gamma * (I[i-1] + dt/2 * m1)
        n2 = gamma * (I[i-1] + dt/2 * m1) - alpha * (R[i-1] + dt/2 * n1)

        S[i] = S[i-1] + dt * k2
        E[i] = E[i-1] + dt * l2
        I[i] = I[i-1] + dt * m2
        R[i] = R[i-1] + dt * n2

    return S, E, I, R


# Example usage

S0 = 990
E0 = 10
I0 = 0
R0 = 0
beta = 0.3
gamma = 0.1
alpha = 0.1
delta = 0.05
t_max = 100
dt = 0.1

S, E, I, R = serid_rk2(S0, E0, I0, R0, beta, gamma, delta, alpha, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SERID Model using RK2')
plt.legend()
plt.show()```

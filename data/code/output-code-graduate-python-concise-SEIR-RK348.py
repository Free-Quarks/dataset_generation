import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, sigma, gamma, s0, e0, i0, r0, t_end, dt):
    t = np.arange(0, t_end, dt)
    n_steps = len(t)
    s = np.zeros(n_steps)
    e = np.zeros(n_steps)
    i = np.zeros(n_steps)
    r = np.zeros(n_steps)
    s[0] = s0
    e[0] = e0
    i[0] = i0
    r[0] = r0

    for n in range(1, n_steps):
        dsdt = -beta * s[n-1] * i[n-1]
        dedt = beta * s[n-1] * i[n-1] - sigma * e[n-1]
        didt = sigma * e[n-1] - gamma * i[n-1]
        drdt = gamma * i[n-1]

        s[n] = s[n-1] + dt * dsdt
        e[n] = e[n-1] + dt * dedt
        i[n] = i[n-1] + dt * didt
        r[n] = r[n-1] + dt * drdt

    return s, e, i, r

# Example usage
beta = 0.25
sigma = 0.1
gamma = 0.05
s0 = 0.99
e0 = 0.01
i0 = 0.0
r0 = 0.0
t_end = 100
dt = 0.1
s, e, i, r = seir_model(beta, sigma, gamma, s0, e0, i0, r0, t_end, dt)

plt.plot(t, s, label='Susceptible')
plt.plot(t, e, label='Exposed')
plt.plot(t, i, label='Infected')
plt.plot(t, r, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()

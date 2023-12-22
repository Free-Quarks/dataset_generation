import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def rk2(y, t, dt, model, beta, gamma):
    k1 = model(y, t, beta, gamma)
    k2 = model([y[i] + dt * k1[i] for i in range(len(y))], t + dt, beta, gamma)
    return [y[i] + (dt / 2) * (k1[i] + k2[i]) for i in range(len(y))]

# initial conditions
S0, I0, R0 = 0.99, 0.01, 0.00
beta, gamma = 0.6, 0.1
t = np.linspace(0, 160, 160)
y0 = [S0, I0, R0]
sol = [y0]
for i in range(1, len(t)):
    sol.append(rk2(sol[-1], t[i-1], t[i]-t[i-1], sir_model, beta, gamma))
sol = np.array(sol)

# Plotting the results
plt.figure(figsize=[6,4])
plt.plot(t, sol[:, 0], label='Susceptible')
plt.plot(t, sol[:, 1], label='Infected')
plt.plot(t, sol[:, 2], label='Recovered')
plt.grid()
plt.legend()
plt.xlabel('Time')
plt.ylabel('Proportions')
plt.title('SIR model with RK2 method')
plt.show()

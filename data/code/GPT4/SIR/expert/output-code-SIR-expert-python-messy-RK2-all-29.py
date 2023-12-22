import numpy as np
import matplotlib.pyplot as plt
import json

# SIR model parameters
beta, gamma = 0.2, 1./10 
N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0
t = np.linspace(0, 160, 160)

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# RK2 method
def rk2(y, t, dt, derivs):
    k0 = dt*derivs(y, t)
    k1 = dt*derivs(y + k0, t + dt)
    y_next = y + 0.5*(k0 + k1)
    return y_next

# Initial conditions vector
y0 = S0, I0, R0

# Empty list to store the solution
soln = [] 
soln.append(y0)

# RK2 method
for i in range(1,len(t)):
    dt = t[i] - t[i-1]
    y_next = rk2(soln[-1], t[i-1], dt, sir_model)
    soln.append(y_next)

S, I, R = np.array(soln).T

# Plot result
plt.figure(figsize=[6,4])
plt.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
plt.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
plt.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered')
plt.xlabel('Time /days')
plt.ylabel('Number (1000s)')
plt.legend()
plt.show()

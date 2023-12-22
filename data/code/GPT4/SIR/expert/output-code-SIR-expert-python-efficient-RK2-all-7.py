import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def rk2_sir(y, t, dt, model, beta, gamma):
    k1 = model(y, t, beta, gamma)
    k2 = model([y[0]+0.5*dt*k1[0],y[1]+0.5*dt*k1[1],y[2]+0.5*dt*k1[2]], t+0.5*dt, beta, gamma)
    y_new = [y[0]+dt*k2[0], y[1]+dt*k2[1], y[2]+dt*k2[2]]
    return y_new

# Initial conditions
S0, I0, R0 = 0.9, 0.1, 0.0  # initial conditions: 90% susceptible, 10% infected, 0% recovered
beta, gamma = 0.2, 0.1  # infection rate/recovery rate

t = 0.0
dt = 0.01
t_end = 100.0
y = [S0,I0,R0]
times = [t]
solution = [y]

while t < t_end:
    t = t + dt
    y = rk2_sir(y, t, dt, sir_model, beta, gamma)
    times.append(t)
    solution.append(y)
S = [y[0] for y in solution]
I = [y[1] for y in solution]
R = [y[2] for y in solution]

plt.figure()
plt.plot(times, S, label='Susceptible')
plt.plot(times, I, label='Infected')
plt.plot(times, R, label='Recovered')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Fraction')
plt.show()

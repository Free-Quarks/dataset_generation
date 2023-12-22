import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return([dS_dt, dI_dt, dR_dt])

def rk3(y, t, dt, model, beta, gamma):
    k1 = model(y, t, beta, gamma)
    k2 = model(y +dt*2/3*k1, t + dt*2/3, beta, gamma)
    k3 = model(y + dt*(1/4*k1 + 3/4*k2), t + dt, beta, gamma)
    y_next = y + dt*(1/4*k1 + 3/8*k2 + 3/8*k3)
    return y_next

def simulate_sir_rk3(S0, I0, R0, beta, gamma, t, dt):
    S, I, R = [S0], [I0], [R0]
    for _ in t[1:]:
        next_S, next_I, next_R = rk3([S[-1], I[-1], R[-1]], _, dt, sir_model, beta, gamma)
        S.append(next_S)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, I, R]).T

# initial conditions
S0, I0, R0 = 0.99, 0.01, 0
beta, gamma = 0.6, 0.1
t = np.linspace(0, 100, 10000)
dt = 100/10000

# simulate model
result = simulate_sir_rk3(S0, I0, R0, beta, gamma, t, dt)

# plot
plt.figure(figsize=[6,4])
plt.plot(t, result)
plt.legend(["S(t)", "I(t)", "R(t)"])
plt.grid()
plt.show()

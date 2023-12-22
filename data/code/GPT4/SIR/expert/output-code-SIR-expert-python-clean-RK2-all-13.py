import numpy as np
import matplotlib.pyplot as plt

# Function to compute the SIR model dynamics
def sir_model_rk2(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta*S*I
    dI_dt = beta*S*I - gamma*I
    dR_dt = gamma*I
    return ([dS_dt, dI_dt, dR_dt])

# RK2 method for integration
def rk2(y, t, dt, model, beta, gamma):
    k1 = model(y, t, beta, gamma)
    k2 = model([y[i] + 0.5*dt*k1[i] for i in range(len(y))], t + 0.5*dt, beta, gamma)
    return [y[i] + dt*k2[i] for i in range(len(y))]

# Initial conditions
S0, I0, R0 = 0.9, 0.1, 0.0
beta, gamma = 0.35, 0.1

# Time vector
t = np.linspace(0, 100, 10000)

# Initialize solution arrays
S, I, R = [S0], [I0], [R0]

# Integrate the equations using RK2
for i in range(1, len(t)):
    dt = t[i] - t[i-1]
    next_S, next_I, next_R = rk2([S[-1], I[-1], R[-1]], t[i-1], dt, sir_model_rk2, beta, gamma)
    S.append(next_S)
    I.append(next_I)
    R.append(next_R)

# Plotting S, I, R over time
plt.figure(figsize = (6, 4))
plt.plot(t, S, label = 'S')
plt.plot(t, I, label = 'I')
plt.plot(t, R, label = 'R')
plt.legend()
plt.show()

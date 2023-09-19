import numpy as np
import matplotlib.pyplot as plt
def sidarthe_model(y, t, beta, sigma, omega, N, theta, phi):
    S, I, D, A, R, T, H, E = y
    dSdt = -beta*S*I/N
    dIdt = beta*S*I/N - sigma*I
    dDdt = phi*sigma*I - omega*D
    dAdt = (1-phi)*sigma*I - theta*A
    dRdt = theta*A
    dTdt = omega*D
    dHdt = phi*sigma*I
    dEdt = sigma*(S*I/N)
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt
def RK3(y0, t0, t1, dt, params, model):
    t = np.arange(t0, t1+dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = dt * model(y[i-1], t[i-1], *params)
        k2 = dt * model(y[i-1] + k1/2, t[i-1] + dt/2, *params)
        k3 = dt * model(y[i-1] + k2*2 - k1, t[i-1] + dt, *params)
        y[i] = y[i-1] + (k1 + 4*k2 + k3)/6
    return t, y

# Initial conditions
N = 1e7
I0, D0, A0, R0, T0, H0, E0 = 1, 0, 0, 0, 0, 0, N-1
S0 = N - I0 - D0 - A0 - R0 - T0 - H0 - E0
y0 = S0, I0, D0, A0, R0, T0, H0, E0

# Parameters
beta = 0.3
sigma = 1/5
omega = 1/10
theta = 0.6
phi = 0.2
params = beta, sigma, omega, N, theta, phi

# Simulate the model
t, y = RK3(y0, 0, 365, 1, params, sidarthe_model)

# Plot the results
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(t, y[:,0], label='Susceptible')
ax.plot(t, y[:,1], label='Infected')
ax.plot(t, y[:,2], label='Deceased')
ax.plot(t, y[:,3], label='Asymptomatic')
ax.plot(t, y[:,4], label='Recovered')
ax.plot(t, y[:,5], label='Cured')
ax.plot(t, y[:,6], label='Hospitalized')
ax.plot(t, y[:,7], label='Exposed')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.set_ylim(0, N)
ax.legend()
plt.show()

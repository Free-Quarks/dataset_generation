import numpy as np
from scipy.integrate import odeint

# function that returns dy/dt

def seir_model(y, t, beta, gamma, sigma):
    # unpack the state variables
    S, E, I, R = y
    N = S + E + I + R
    # compute the derivatives
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# initial condition
y0 = S0, E0, I0, R0
# time points
t = np.linspace(0, 160, 160)

# parameters
gamma = 1/14
sigma = 1/5
beta = 1.75

# solve ODE
sol = odeint(seir_model, y0, t, args=(beta, gamma, sigma))

# plot results
plt.figure(figsize=(10,6))
plt.plot(t, sol[:, 0], 'b', label='Susceptible')
plt.plot(t, sol[:, 1], 'y', label='Exposed')
plt.plot(t, sol[:, 2], 'r', label='Infected')
plt.plot(t, sol[:, 3], 'g', label='Recovered')
plt.xlabel('Time')
plt.ylabel('Fraction of population')
plt.title('SEIR Model')
plt.legend(loc='best')
plt.show()

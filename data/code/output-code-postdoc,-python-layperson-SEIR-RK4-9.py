import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    def deriv(y, t, beta, gamma, sigma, N):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    t = np.linspace(0, T, T)

    y0 = N - I0 - E0 - R0
    y = S0, E0, I0, R0

    ret = odeint(deriv, y, t, args=(beta, gamma, sigma, N))
    S, E, I, R = ret.T

    return t, S, E, I, R


beta = 0.2
sigma = 0.1
gamma = 0.1
N = 1000
I0 = 1
E0 = 0
R0 = 0
T = 160


# Run the model
t, S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, T)


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SEIR Model')
plt.legend()
plt.grid(True)
plt.show()

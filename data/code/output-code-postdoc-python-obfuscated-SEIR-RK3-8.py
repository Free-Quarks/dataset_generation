import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, sigma, gamma, N, I0, E0, R0, T):
    def derivs(y, t, beta, sigma, gamma, N):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    t = np.arange(0, T, 1)

    y0 = N - I0, E0, I0, R0
    ret = np.zeros((T, 4))
    ret[0] = y0

    for i in range(1, T):
        tspan = [t[i - 1], t[i]]
        y = np.integrate.odeint(derivs, y0, tspan, args=(beta, sigma, gamma, N))
        ret[i] = y[-1]
        y0 = y[-1]

    S, E, I, R = ret.T

    return S, E, I, R


# Parameters
beta = 0.2
sigma = 1/5
gamma = 1/10
N = 1000
I0, E0, R0 = 1, 0, 0
T = 160

# Run model
S, E, I, R = seir_model(beta, sigma, gamma, N, I0, E0, R0, T)

# Plotting
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SEIR Model')
plt.show()


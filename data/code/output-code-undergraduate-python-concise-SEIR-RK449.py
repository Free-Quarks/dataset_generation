import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, I0, R0, N, T):
    def derivs(y, t):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    t = np.linspace(0, T, T)
    y0 = N - I0 - R0, I0, I0, R0
    ode_result = spi.odeint(derivs, y0, t)
    S, E, I, R = ode_result.T

    plt.figure(figsize=(12, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()


seir_model(beta=0.2, gamma=0.1, sigma=0.5, I0=1, R0=0, N=1000, T=160)

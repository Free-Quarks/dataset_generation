import numpy as np
import matplotlib.pyplot as plt


def seir_model(R0, alpha, beta, gamma, N, I0, E0, days):
    def deriv(y, t, N, R0, alpha, beta, gamma):
        S, E, I, R = y
        dSdt = - (R0 / alpha) * beta * S * I / N
        dEdt = (R0 / alpha) * beta * S * I / N - alpha * E
        dIdt = alpha * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    t = np.linspace(0, days, days)
    y0 = N - I0, E0, I0, 0
    ret = odeint(deriv, y0, t, args=(N, R0, alpha, beta, gamma))
    S, E, I, R = ret.T

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()


seir_model(2.5, 1/5.2, 1/2.9, 1/10, 1000, 1, 0, 160)

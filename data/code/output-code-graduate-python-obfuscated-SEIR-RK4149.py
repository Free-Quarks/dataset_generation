import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, sigma, gamma, N, I0, E0, R0, T):
    def deriv(y, t, beta, sigma, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt
    
    t = np.linspace(0, T, T)
    y0 = N - I0, E0, I0, R0
    ret = odeint(deriv, y0, t, args=(beta, sigma, gamma))
    S, E, I, R = ret.T
    
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, E, 'y', label='Exposed')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()
}

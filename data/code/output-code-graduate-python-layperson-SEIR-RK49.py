import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, gamma, sigma, population, infected, exposed, recovered, days):
    N = population
    I0, E0, R0 = infected, exposed, recovered
    S0 = N - I0 - R0 - E0
    beta, gamma, sigma = beta, gamma, sigma
    t = np.linspace(0, days, days)
    
    def deriv(y, t, N, beta, gamma, sigma):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt
    
    y0 = S0, E0, I0, R0
    
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = ret.T
    
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, E, 'y', label='Exposed')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()
}


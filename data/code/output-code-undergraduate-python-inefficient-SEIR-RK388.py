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
    y0 = S0, E0, I0, R0
    ret = odeint(deriv, y0, t, args=(beta, gamma, sigma, N))
    S, E, I, R = ret.T
    
    plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    plt.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
    plt.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    plt.legend(loc='best')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.grid(True)
    plt.show()
    
    return S, E, I, R


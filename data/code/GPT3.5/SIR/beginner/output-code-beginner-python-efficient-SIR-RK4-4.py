import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, days):
    # Total population size
    S0 = N - I0 - R0
    # Initial conditions
    y0 = [S0, I0, R0]
    # Time vector
    t = np.linspace(0, days, days)
    
    def deriv(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    # Solve the ODE system
    ret = odeint(deriv, y0, t, args=(beta, gamma))
    S, I, R = ret.T
    
    # Plot the results
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()
}


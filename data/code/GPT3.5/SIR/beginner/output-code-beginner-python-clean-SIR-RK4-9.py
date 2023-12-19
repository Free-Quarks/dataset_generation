import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, population, initial_infected, time):
    # Define the differential equations
    def deriv(y, t):
        S, I, R = y
        dSdt = -beta * S * I / population
        dIdt = beta * S * I / population - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    # Set up initial conditions
    initial_susceptible = population - initial_infected
    y0 = initial_susceptible, initial_infected, 0
    
    # Set up time grid
    t = np.linspace(0, time, time+1)
    
    # Integrate the differential equations
    sol = odeint(deriv, y0, t)
    S, I, R = sol.T
    
    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()
}


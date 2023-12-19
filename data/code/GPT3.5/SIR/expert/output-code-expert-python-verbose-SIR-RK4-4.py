import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, S0, I0, R0, t)

    # Define the differential equations
    def deriv(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Set initial conditions
    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid
    ret = odeint(deriv, y0, t, args=(beta, gamma))
    S, I, R = ret.T

    # Plot the data
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, S, 'b', alpha=0.7, label='Susceptible')
    plt.plot(t, I, 'r', alpha=0.7, label='Infected')
    plt.plot(t, R, 'g', alpha=0.7, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()


# Set the model parameters
beta = 0.2
# transmission rate

gamma = 0.1
# recovery rate

S0 = 1000
# initial number of susceptible individuals

I0 = 1
# initial number of infected individuals

R0 = 0
# initial number of recovered individuals

t = np.linspace(0, 160, 160)
# time grid

sir_model(beta, gamma, S0, I0, R0, t)

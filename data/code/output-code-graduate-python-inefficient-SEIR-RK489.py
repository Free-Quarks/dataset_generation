import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, population, I0, E0, R0, days):
    # Function to implement the SEIR model
    def model(y, t):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Initial conditions
    S0 = population - (I0 + E0 + R0)
    y0 = [S0, E0, I0, R0]
    
    # Time points
    t = np.linspace(0, days, days)
    
    # Solve the differential equations
    sol = integrate.odeint(model, y0, t)
    
    # Return the solution
    return sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

# Parameters
beta = 0.2
sigma = 0.1
gamma = 0.1
population = 1000000
I0 = 1000
E0 = 1000
R0 = 0
days = 100

# Call the SEIR model
S, E, I, R = seir_model(beta, gamma, sigma, population, I0, E0, R0, days)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.title('SEIR Model')
plt.show()

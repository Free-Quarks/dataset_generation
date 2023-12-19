import numpy as np
import matplotlib.pyplot as plt


# Function to simulate SIR model using Euler's method

def sir_model(beta, gamma, population, initial_infected, initial_recovered, t_max, dt):
    
    # Initialize arrays to store time, susceptible, infected, and recovered population
    t = np.arange(0, t_max+dt, dt)
    s = np.zeros(len(t))
    i = np.zeros(len(t))
    r = np.zeros(len(t))
    
    # Set initial values
    s[0] = population - initial_infected - initial_recovered
    i[0] = initial_infected
    r[0] = initial_recovered
    
    # Euler's method iteration
    for j in range(len(t)-1):
        s[j+1] = s[j] - beta * s[j] * i[j] * dt / population
        i[j+1] = i[j] + (beta * s[j] * i[j] - gamma * i[j]) * dt / population
        r[j+1] = r[j] + gamma * i[j] * dt / population
    
    # Return the simulation results
    return (t, s, i, r)


# Set the model parameters
beta = 0.2
# Infection rate
gamma = 0.1
# Recovery rate
population = 1000
# Total population size
initial_infected = 10
# Initial infected population
initial_recovered = 0
# Initial recovered population
t_max = 100
# Maximum simulation time
dt = 0.1
# Time step size


# Call the function to simulate the SIR model
(t, s, i, r) = sir_model(beta, gamma, population, initial_infected, initial_recovered, t_max, dt)


# Plot the simulation results
plt.plot(t, s, label='Susceptible')
plt.plot(t, i, label='Infected')
plt.plot(t, r, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()

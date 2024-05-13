import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, sigma, gamma, population, initial_infected, incubation_period, recovery_time, duration):
    # Calculate the number of time steps
    num_steps = int(duration)
    
    # Create arrays to store the population numbers
    susceptible = np.zeros(num_steps)
    exposed = np.zeros(num_steps)
    infected = np.zeros(num_steps)
    recovered = np.zeros(num_steps)
    
    # Set initial conditions
    susceptible[0] = population - initial_infected
    exposed[0] = 0
    infected[0] = initial_infected
    recovered[0] = 0
    
    # Define the time step size
    dt = 1.0
    
    # Run the simulation
    for i in range(1, num_steps):
        # Compute the derivatives
        dS = -beta * susceptible[i-1] * infected[i-1] / population
        dE = beta * susceptible[i-1] * infected[i-1] / population - sigma * exposed[i-1]
        dI = sigma * exposed[i-1] - gamma * infected[i-1]
        dR = gamma * infected[i-1]
        
        # Update the population numbers
        susceptible[i] = susceptible[i-1] + dt * dS
        exposed[i] = exposed[i-1] + dt * dE
        infected[i] = infected[i-1] + dt * dI
        recovered[i] = recovered[i-1] + dt * dR
        
    # Return the simulation results
    return susceptible, exposed, infected, recovered


# Example usage
beta = 0.8
sigma = 1/5
gamma = 1/10
population = 100000
initial_infected = 10
incubation_period = 5
recovery_time = 10
duration = 100

susceptible, exposed, infected, recovered = seir_model(beta, sigma, gamma, population, initial_infected, incubation_period, recovery_time, duration)

# Plot the results
plt.plot(susceptible, label='Susceptible')
plt.plot(exposed, label='Exposed')
plt.plot(infected, label='Infected')
plt.plot(recovered, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

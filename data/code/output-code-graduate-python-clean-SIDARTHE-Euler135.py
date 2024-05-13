import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(population, initial_conditions, parameters, timesteps):
    # Unpack initial conditions
    susceptible_0, infected_0, exposed_0, hospitalized_0, critical_0, recovered_0, dead_0 = initial_conditions
    
    # Unpack parameters
    beta, sigma, theta, delta, gamma_h, gamma_c, gamma_r, gamma_d = parameters
    
    # Create arrays to store the number of individuals in each compartment at each timestep
    susceptible = np.zeros(timesteps)
    infected = np.zeros(timesteps)
    exposed = np.zeros(timesteps)
    hospitalized = np.zeros(timesteps)
    critical = np.zeros(timesteps)
    recovered = np.zeros(timesteps)
    dead = np.zeros(timesteps)
    
    # Set initial conditions
    susceptible[0] = susceptible_0
    infected[0] = infected_0
    exposed[0] = exposed_0
    hospitalized[0] = hospitalized_0
    critical[0] = critical_0
    recovered[0] = recovered_0
    dead[0] = dead_0
    
    # Iterate over timesteps
    for t in range(1, timesteps):
        # Compute the flows between compartments
        flow_susceptible_infected = beta * susceptible[t-1] * infected[t-1] / population
        flow_exposed_infected = sigma * exposed[t-1]
        flow_exposed_hospitalized = theta * exposed[t-1]
        flow_infected_hospitalized = delta * infected[t-1]
        flow_infected_critical = gamma_h * infected[t-1]
        flow_infected_recovered = gamma_c * infected[t-1]
        flow_hospitalized_critical = gamma_r * hospitalized[t-1]
        flow_critical_dead = gamma_d * critical[t-1]
        
        # Update the number of individuals in each compartment
        susceptible[t] = susceptible[t-1] - flow_susceptible_infected
        infected[t] = infected[t-1] + flow_susceptible_infected - flow_exposed_infected - flow_exposed_hospitalized
        exposed[t] = exposed[t-1] + flow_exposed_infected - flow_infected_hospitalized
        hospitalized[t] = hospitalized[t-1] + flow_exposed_hospitalized - flow_infected_critical - flow_hospitalized_critical
        critical[t] = critical[t-1] + flow_infected_critical - flow_critical_dead
        recovered[t] = recovered[t-1] + flow_infected_recovered + flow_hospitalized_critical
        dead[t] = dead[t-1] + flow_critical_dead
    
    # Return the arrays of compartments
    return susceptible, infected, exposed, hospitalized, critical, recovered, dead
}

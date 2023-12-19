import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(initial_conditions, parameters, t)
    # Unpack initial conditions
    S0, I0, R0 = initial_conditions
    # Unpack parameters
    beta, gamma = parameters
    
    # Set up arrays to store the results
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    # Initialize the arrays with the initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Calculate the time step
    dt = t[1] - t[0]
    
    # Iterate over each time step
    for i in range(1, len(t)):
        # Calculate the derivatives
        dSdt = -beta * S[i-1] * I[i-1]
        dIdt = beta * S[i-1] * I[i-1] - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        
        # Calculate the intermediate values
        S_star = S[i-1] + 0.5 * dt * dSdt
        I_star = I[i-1] + 0.5 * dt * dIdt
        R_star = R[i-1] + 0.5 * dt * dRdt
        
        # Calculate the derivatives using the intermediate values
        dSdt_star = -beta * S_star * I_star
        dIdt_star = beta * S_star * I_star - gamma * I_star
        dRdt_star = gamma * I_star
        
        # Update the values using the intermediate derivatives
        S[i] = S[i-1] + dt * dSdt_star
        I[i] = I[i-1] + dt * dIdt_star
        R[i] = R[i-1] + dt * dRdt_star
        
    # Return the results
    return S, I, R


# Define the initial conditions
initial_conditions = (990, 10, 0)

# Define the parameters
parameters = (0.3, 0.1)

# Define the time array
t = np.linspace(0, 100, 100)

# Run the simulation
S, I, R = SIR_RK2(initial_conditions, parameters, t)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()

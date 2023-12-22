import numpy as np
import matplotlib.pyplot as plt
import json

def SIR_model(S, I, R, beta, gamma, dt):
    """
    Simulate the SIR epidemic model using the Euler method.

    Parameters
    ----------
    S : float
        The initial number of susceptible individuals.
    I : float
        The initial number of infected individuals.
    R : float
        The initial number of recovered individuals.
    beta : float
        The infection rate.
    gamma : float
        The recovery rate.
    dt : float
        The time step.

    Returns
    -------
    S_list, I_list, R_list : list of float
        The time series of the S, I, R compartments.

    """
    # Create lists to store the S, I, R values
    S_list = []
    I_list = []
    R_list = []

    # Append the initial values
    S_list.append(S)
    I_list.append(I)
    R_list.append(R)

    # Start the simulation
    t = 0
    while t < 100:
        # Update the S, I, R values
        S_new = S - beta*S*I*dt
        I_new = I + (beta*S*I - gamma*I)*dt
        R_new = R + gamma*I*dt

        # Append the new values
        S_list.append(S_new)
        I_list.append(I_new)
        R_list.append(R_new)

        # Update the S, I, R values for the next iteration
        S = S_new
        I = I_new
        R = R_new

        # Update the time
        t += dt

    # Return the lists
    return S_list, I_list, R_list


# Initial conditions
S0 = 0.99
I0 = 0.01
R0 = 0.0

# Parameters
beta = 0.5
gamma = 0.1
dt = 0.1

# Run the model
S, I, R = SIR_model(S0, I0, R0, beta, gamma, dt)

# Plot the results
plt.figure(figsize=(6, 4))
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('SIR model')
plt.show()

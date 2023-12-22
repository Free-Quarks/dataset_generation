import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_euler(S0, I0, R0, beta, gamma, n_days):
    """
    A simple SIR model for the spread of diseases using Euler's method.

    Parameters:
    S0 (int): Initial population size.
    I0 (int): Initial number of infected individuals.
    R0 (int): Initial number of recovered individuals.
    beta (float): The parameter controlling how often a susceptible-infected contact results in a new infection.
    gamma (float): The rate an infected recovers and moves into the resistant phase.
    n_days (int): The number of days.

    Returns:
    tuple: Model solution as three numpy arrays (S, I, R)
    """

    # Initialize arrays
    S, I, R = [S0], [I0], [R0]

    for _ in range(n_days):
        next_S = S[-1] - (beta*S[-1]*I[-1])
        next_I = I[-1] + (beta*S[-1]*I[-1] - gamma*I[-1])
        next_R = R[-1] + (gamma*I[-1])

        S.append(next_S)
        I.append(next_I)
        R.append(next_R)

    return np.array(S), np.array(I), np.array(R)

# Test the function
S, I, R = sir_model_euler(999, 1, 0, 0.3, 0.1, 160)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(S, 'b', label='Susceptible')
plt.plot(I, 'r', label='Infected')
plt.plot(R, 'g', label='Recovered')
plt.legend()
plt.title('SIR Model using Euler Method')
plt.xlabel('Days')
plt.ylabel('Population')
plt.show()

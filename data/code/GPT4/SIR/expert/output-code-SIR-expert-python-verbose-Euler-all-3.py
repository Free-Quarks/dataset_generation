import numpy as np
import matplotlib.pyplot as plt
import json

def model_SIR_Euler(beta, gamma, S0, I0, R0, T, dt):
    '''
    Function to simulate the SIR model using Euler's method

    Parameters:

    beta: float
        The parameter controlling how often a susceptible-infected contact results in a new infection.
    gamma: float
        The rate an infected recovers and moves into the resistant phase.
    S0: int
        The initial number of susceptible individuals.
    I0: int
        The initial number of infected individuals.
    R0: int
        The initial number of recovered individuals.
    T: int
        The total time to run the simulation for.
    dt: float
        The time step size.

    Returns:

    S: array
        Array of the number of susceptible individuals at each time step.
    I: array
        Array of the number of infected individuals at each time step.
    R: array
        Array of the number of recovered individuals at each time step.
    time: array
        Array of the time steps.
    '''

    # Initialize variables
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    time = np.arange(0, T+dt, dt)

    # Perform the simulation
    for t in time[1:]:
        dS = -beta*S[-1]*I[-1]/N*dt
        dI = beta*S[-1]*I[-1]/N*dt - gamma*I[-1]*dt
        dR = gamma*I[-1]*dt
        S.append(S[-1] + dS)
        I.append(I[-1] + dI)
        R.append(R[-1] + dR)

    return S, I, R, time.tolist()

#Usage
S, I, R, time = model_SIR_Euler(0.1, 0.02, 1000, 1, 0, 150, 0.1)

#Plotting
plt.figure(figsize=(12,8))
plt.plot(time, S, label='Susceptible')
plt.plot(time, I, label='Infected')
plt.plot(time, R, label='Recovered')
plt.legend()
plt.title('SIR Model using Euler')
plt.xlabel('Time')
plt.ylabel('Population')
plt.grid()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import json

def RK3_SIR(S0, I0, R0, beta, gamma, t):
    
    This function simulates the SIR model using the RK3 method.

    Parameters:
    S0 (float): the initial number of susceptible individuals.
    I0 (float): the initial number of infected individuals.
    R0 (float): the initial number of recovered individuals.
    beta (float): The parameter controlling how often a susceptible-infected contact results in a new infection.
    gamma (float): The rate an infected recovers and moves into the resistant phase.
    t (float): the time period.

    Returns:
    S (list): List of susceptible individuals over time t.
    I (list): List of infected individuals over time t.
    R (list): List of recovered individuals over time t.

    S, I, R = [S0], [I0], [R0]

    dt = 0.01
    t = np.arange(0, t, dt)

    for _ in t:
        next_S = S[-1] - dt * beta * S[-1] * I[-1]
        next_I = I[-1] + dt * (beta * S[-1] * I[-1] - gamma * I[-1])
        next_R = R[-1] + dt * gamma * I[-1]

        S.append(next_S)
        I.append(next_I)
        R.append(next_R)

    plt.figure(figsize=[6,4])
    plt.plot(t, S[1:], label='Susceptible')
    plt.plot(t, I[1:], label='Infected')
    plt.plot(t, R[1:], label='Recovered')
    plt.legend()
    plt.grid()
    plt.show()

code = RK3_SIR.__code__
docstring = RK3_SIR.__doc__

output = {
    'code': str(code),
    'function_name': RK3_SIR.__name__
}

output_json = json.dumps(output)

print(output_json)

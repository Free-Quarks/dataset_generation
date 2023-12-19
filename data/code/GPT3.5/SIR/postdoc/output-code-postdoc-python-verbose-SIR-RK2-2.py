import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(N, beta, gamma, I0, R0, T):
    """
    Simulate the SIR model using the Runge-Kutta 2nd order method
    """
    # Calculate the initial values
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    dt = 0.1
    num_steps = int(T / dt)

    # Run the simulation
    for step in range(num_steps):
        # Calculate the derivatives
        dSdt = -beta * S[-1] * I[-1] / N
        dIdt = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dRdt = gamma * I[-1]

        # Calculate the intermediate values
        S_intermediate = S[-1] + 0.5 * dt * dSdt
        I_intermediate = I[-1] + 0.5 * dt * dIdt
        R_intermediate = R[-1] + 0.5 * dt * dRdt

        # Calculate the derivatives at the intermediate values
        dSdt_intermediate = -beta * S_intermediate * I_intermediate / N
        dIdt_intermediate = beta * S_intermediate * I_intermediate / N - gamma * I_intermediate
        dRdt_intermediate = gamma * I_intermediate

        # Calculate the next values
        S_next = S[-1] + dt * dSdt_intermediate
        I_next = I[-1] + dt * dIdt_intermediate
        R_next = R[-1] + dt * dRdt_intermediate

        # Append the next values to the lists
        S.append(S_next)
        I.append(I_next)
        R.append(R_next)

    return S, I, R


# Set the parameters
N = 1000
beta = 0.2
gamma = 0.1
I0 = 1
R0 = 0
T = 10

# Run the simulation
S, I, R = SIR_RK2(N, beta, gamma, I0, R0, T)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()


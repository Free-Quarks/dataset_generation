import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, N, I0, R0, T):
    # Total population size
    S0 = N - I0 - R0
    # Time step size
    dt = 0.1
    # Number of steps
    num_steps = int(T / dt)
    # Initialize arrays
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    t = np.zeros(num_steps)
    # Initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    t[0] = 0
    # Runge-Kutta method
    for i in range(1, num_steps):
        # Compute derivatives
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        # Update values using RK2
        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt
        t[i] = t[i-1] + dt
    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()
}


import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, N, I0, R0, T):
    # Parameters
    dt = 0.1
    num_steps = int(T/dt)
    t = np.linspace(0, T, num_steps + 1)

    # Arrays
    S = np.zeros(num_steps + 1)
    I = np.zeros(num_steps + 1)
    R = np.zeros(num_steps + 1)

    # Initial conditions
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    # Runge-Kutta method
    for i in range(num_steps):
        # Calculate derivatives
        dSdt = -beta*S[i]*I[i]/N
        dIdt = beta*S[i]*I[i]/N - gamma*I[i]
        dRdt = gamma*I[i]

        # Update variables
        S[i+1] = S[i] + dt*dSdt
        I[i+1] = I[i] + dt*dIdt
        R[i+1] = R[i] + dt*dRdt

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()


# Example usage
SIR_RK2(beta=0.2, gamma=0.1, N=1000, I0=1, R0=0, T=100)

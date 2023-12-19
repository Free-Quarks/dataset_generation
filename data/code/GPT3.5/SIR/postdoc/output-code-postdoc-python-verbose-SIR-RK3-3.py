import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(S0, I0, R0, beta, gamma, t_max, N, h):
    # Define the derivative functions for the SIR model
    def dS_dt(S, I, beta):
        return -beta * S * I / N

    def dI_dt(S, I, beta, gamma):
        return beta * S * I / N - gamma * I

    def dR_dt(I, gamma):
        return gamma * I

    # Initialize arrays to store the values
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)

    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0

    # Runge-Kutta integration
    for t in range(t_max - 1):
        k1_S = dS_dt(S[t], I[t], beta)
        k1_I = dI_dt(S[t], I[t], beta, gamma)
        k1_R = dR_dt(I[t], gamma)

        S_mid = S[t] + 0.5 * h * k1_S
        I_mid = I[t] + 0.5 * h * k1_I
        R_mid = R[t] + 0.5 * h * k1_R

        k2_S = dS_dt(S_mid, I_mid, beta)
        k2_I = dI_dt(S_mid, I_mid, beta, gamma)
        k2_R = dR_dt(I_mid, gamma)

        S_new = S[t] + h * k2_S
        I_new = I[t] + h * k2_I
        R_new = R[t] + h * k2_R

        S[t + 1] = S_new
        I[t + 1] = I_new
        R[t + 1] = R_new

    # Plotting
    plt.plot(S, label='Susceptible')
    plt.plot(I, label='Infected')
    plt.plot(R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SIR Model Simulation')
    plt.show()


SIR_RK3(999, 1, 0, 0.3, 0.1, 100, 1000, 0.1)

import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, t_end):
    S0 = N - I0 - R0
    dt = 0.1
    num_steps = int(t_end / dt)
    t = np.linspace(0, t_end, num_steps+1)
    S = np.zeros(num_steps+1)
    I = np.zeros(num_steps+1)
    R = np.zeros(num_steps+1)

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(num_steps):
        dSdt = -beta * S[i] * I[i] / N
        dIdt = beta * S[i] * I[i] / N - gamma * I[i]
        dRdt = gamma * I[i]

        S_half = S[i] + 0.5 * dt * dSdt
        I_half = I[i] + 0.5 * dt * dIdt
        R_half = R[i] + 0.5 * dt * dRdt

        dSdt_half = -beta * S_half * I_half / N
        dIdt_half = beta * S_half * I_half / N - gamma * I_half
        dRdt_half = gamma * I_half

        S[i+1] = S[i] + dt * dSdt_half
        I[i+1] = I[i] + dt * dIdt_half
        R[i+1] = R[i] + dt * dRdt_half

    return t, S, I, R


def plot_SIR(t, S, I, R):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Example usage
N = 10000
I0 = 100
R0 = 0
beta = 0.2
gamma = 0.1
t_end = 100

# Run SIR model
t, S, I, R = SIR_model(beta, gamma, N, I0, R0, t_end)

# Plot results
plot_SIR(t, S, I, R)

import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt):
    # Define the derivative function
    def derivative(S, I, R, beta, gamma):
        dS_dt = -beta * S * I
        dI_dt = beta * S * I - gamma * I
        dR_dt = gamma * I
        return dS_dt, dI_dt, dR_dt

    # Initialize arrays
    t = np.arange(0, t_max, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0

    # Perform RK4 integration
    for i in range(1, len(t)):
        dS1, dI1, dR1 = derivative(S[i-1], I[i-1], R[i-1], beta, gamma)
        dS2, dI2, dR2 = derivative(S[i-1] + dt/2 * dS1, I[i-1] + dt/2 * dI1, R[i-1] + dt/2 * dR1, beta, gamma)
        dS3, dI3, dR3 = derivative(S[i-1] + dt/2 * dS2, I[i-1] + dt/2 * dI2, R[i-1] + dt/2 * dR2, beta, gamma)
        dS4, dI4, dR4 = derivative(S[i-1] + dt * dS3, I[i-1] + dt * dI3, R[i-1] + dt * dR3, beta, gamma)
        S[i] = S[i-1] + dt/6 * (dS1 + 2*dS2 + 2*dS3 + dS4)
        I[i] = I[i-1] + dt/6 * (dI1 + 2*dI2 + 2*dI3 + dI4)
        R[i] = R[i-1] + dt/6 * (dR1 + 2*dR2 + 2*dR3 + dR4)

    # Return the arrays
    return t, S, I, R


def plot_SIR(t, S, I, R):
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()


t_max = 100
dt = 0.1
S0 = 999
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1

# Run simulation
t, S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt)

# Plot the results
plot_SIR(t, S, I, R)

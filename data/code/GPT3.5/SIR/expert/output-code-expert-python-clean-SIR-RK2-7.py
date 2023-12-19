import numpy as np
import matplotlib.pyplot as plt


# Function to implement SIR model using RK2

def sir_model_rk2(N, I0, R0, beta, gamma, t_max, dt):
    # Initialize arrays
    t = np.arange(0, t_max, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    # Set initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = R0

    # Runge-Kutta 2nd order
    for i in range(1, len(t)):
        k1 = dt * (beta * S[i-1] * I[i-1] / N)
        l1 = dt * (gamma * I[i-1])
        k2 = dt * (beta * (S[i-1] - 0.5 * k1) * (I[i-1] - 0.5 * l1) / N)
        l2 = dt * (gamma * (I[i-1] - 0.5 * l1))

        S[i] = S[i-1] - k2
        I[i] = I[i-1] + k2 - l2
        R[i] = R[i-1] + l2

    return S, I, R


# Parameters
N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_max = 100
dt = 0.1

# Run the model
S, I, R = sir_model_rk2(N, I0, R0, beta, gamma, t_max, dt)

# Plot the results
plt.figure()
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time Steps')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()


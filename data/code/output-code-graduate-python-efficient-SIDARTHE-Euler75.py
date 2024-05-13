import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(params, initial_conditions, t_start, t_end, dt):
    # Unpack parameters
    alpha, beta, gamma, delta, theta, epsilon = params
    # Unpack initial conditions
    S0, I0, D0, A0, R0, T0, H0, E0 = initial_conditions
    # Initialize arrays
    t = np.arange(t_start, t_end, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    D = np.zeros_like(t)
    A = np.zeros_like(t)
    R = np.zeros_like(t)
    T = np.zeros_like(t)
    H = np.zeros_like(t)
    E = np.zeros_like(t)
    # Set initial values
    S[0] = S0
    I[0] = I0
    D[0] = D0
    A[0] = A0
    R[0] = R0
    T[0] = T0
    H[0] = H0
    E[0] = E0
    # Euler's method
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1]
        dI = beta * S[i-1] * I[i-1] - (alpha + gamma) * I[i-1] - delta * D[i-1] - theta * A[i-1]
        dD = delta * D[i-1] + epsilon * T[i-1]
        dA = theta * A[i-1] - epsilon * T[i-1]
        dR = gamma * I[i-1]
        dT = alpha * I[i-1] + alpha * H[i-1] - epsilon * T[i-1]
        dH = epsilon * T[i-1] - alpha * H[i-1]
        dE = epsilon * T[i-1]
        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        D[i] = D[i-1] + dt * dD
        A[i] = A[i-1] + dt * dA
        R[i] = R[i-1] + dt * dR
        T[i] = T[i-1] + dt * dT
        H[i] = H[i-1] + dt * dH
        E[i] = E[i-1] + dt * dE
    # Return results
    return S, I, D, A, R, T, H, E


# Example usage
params = (0.2, 0.1, 0.05, 0.05, 0.1, 0.1)
initial_conditions = (99, 1, 0, 0, 0, 0, 0, 0)
t_start = 0
t_end = 100
dt = 0.1

S, I, D, A, R, T, H, E = sidarthe_model(params, initial_conditions, t_start, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(D, label='Deceased')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Tested')
plt.plot(H, label='Hospitalized')
plt.plot(E, label='Exposed')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIDARTHE Model')
plt.show()

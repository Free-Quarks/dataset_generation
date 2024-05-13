import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(beta, epsilon, delta, theta, alpha, rho, N, I0, D0, A0, R0, T, dt):
    # Initialize arrays
    t = np.linspace(0, T, int(T/dt) + 1)
    S = np.zeros(t.shape)
    I = np.zeros(t.shape)
    D = np.zeros(t.shape)
    A = np.zeros(t.shape)
    R = np.zeros(t.shape)
    H = np.zeros(t.shape)
    T = np.zeros(t.shape)
    E = np.zeros(t.shape)
    Q = np.zeros(t.shape)
    M = np.zeros(t.shape)
    C = np.zeros(t.shape)
    S[0] = N - I0 - D0 - A0 - R0 - E[0] - Q[0] - M[0] - C[0]
    I[0] = I0
    D[0] = D0
    A[0] = A0
    R[0] = R0
    H[0] = 0
    T[0] = 0
    E[0] = 0
    Q[0] = 0
    M[0] = 0
    C[0] = 0
    for i in range(1, t.shape[0]):
        S[i] = S[i-1] - (beta * S[i-1] * I[i-1]) * dt
        E[i] = E[i-1] + (beta * S[i-1] * I[i-1] - epsilon * E[i-1]) * dt
        I[i] = I[i-1] + (epsilon * E[i-1] - delta * I[i-1] - theta * I[i-1] - alpha * I[i-1]) * dt
        D[i] = D[i-1] + (alpha * I[i-1]) * dt
        A[i] = A[i-1] + (theta * I[i-1] - rho * A[i-1]) * dt
        R[i] = R[i-1] + (delta * I[i-1] + rho * A[i-1]) * dt
        H[i] = delta * I[i-1] * dt
        T[i] = T[i-1] + (delta * I[i-1] + alpha * I[i-1]) * dt
        Q[i] = (delta * I[i-1] + alpha * I[i-1]) * dt
        M[i] = alpha * I[i-1] * dt
        C[i] = rho * A[i-1] * dt
    return t, S, E, I, D, A, R, H, T, Q, M, C


# Parameters
beta = 0.25
epsilon = 0.1
theta = 0.1
alpha = 0.2
rho = 0.3
N = 100000
I0 = 1000
D0 = 100
A0 = 100
R0 = 100
T = 365
dt = 0.1

# Run the model
t, S, E, I, D, A, R, H, T, Q, M, C = sidarthe_model(beta, epsilon, delta, theta, alpha, rho, N, I0, D0, A0, R0, T, dt)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Deceased')
plt.plot(t, A, label='Asymptomatic')
plt.plot(t, R, label='Recovered')
plt.plot(t, H, label='Hospitalized')
plt.plot(t, T, label='Total')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIDARTHE Model')
plt.show()

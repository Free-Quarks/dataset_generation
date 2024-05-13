import numpy as np
import matplotlib.pyplot as plt


# Function to implement the SIDARTHE model using RK2

def s_idarthe_rk2_model(N, T, dt, alpha, beta, gamma, delta, epsilon, theta, rho, sigma):
    # Initialize arrays to store the results
    S = np.zeros(T)
    I = np.zeros(T)
    D = np.zeros(T)
    A = np.zeros(T)
    R = np.zeros(T)
    T = np.zeros(T)
    H = np.zeros(T)
    E = np.zeros(T)

    # Set initial conditions
    S[0] = N - 1
    I[0] = 1

    # Run the model
    for t in range(1, T):
        # Calculate derivatives using RK2
        k1 = alpha * S[t-1] * I[t-1] / N
        m1 = beta * I[t-1]
        n1 = gamma * I[t-1]
        o1 = delta * I[t-1]
        p1 = epsilon * I[t-1]
        q1 = theta * I[t-1]
        r1 = rho * I[t-1]

        k2 = alpha * (S[t-1] - dt * k1/2) * (I[t-1] - dt * m1/2) / N
        m2 = beta * (I[t-1] - dt * m1/2)
        n2 = gamma * (I[t-1] - dt * n1/2)
        o2 = delta * (I[t-1] - dt * o1/2)
        p2 = epsilon * (I[t-1] - dt * p1/2)
        q2 = theta * (I[t-1] - dt * q1/2)
        r2 = rho * (I[t-1] - dt * r1/2)

        # Update the variables
        S[t] = S[t-1] - dt * (k1 + k2)
        I[t] = I[t-1] - dt * (m1 + m2)
        D[t] = D[t-1] + dt * (n1 + n2)
        A[t] = A[t-1] + dt * (o1 + o2)
        R[t] = R[t-1] + dt * (p1 + p2)
        T[t] = T[t-1] + dt * (q1 + q2)
        H[t] = H[t-1] + dt * (r1 + r2)
        E[t] = E[t-1] + dt * (sigma * I[t-1] * (1 - (I[t-1]/N)))

    # Return the results
    return S, I, D, A, R, T, H, E


# Example usage
N = 100000
T = 100
dt = 1
alpha = 0.1
beta = 0.02
gamma = 0.01
delta = 0.005
epsilon = 0.005
theta = 0.002
rho = 0.001
sigma = 0.1

S, I, D, A, R, T, H, E = s_idarthe_rk2_model(N, T, dt, alpha, beta, gamma, delta, epsilon, theta, rho, sigma)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(D, label='Deaths')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Tested')
plt.plot(H, label='Hospitalized')
plt.plot(E, label='Exposed')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIDARTHE Model using RK2')
plt.legend()
plt.show()


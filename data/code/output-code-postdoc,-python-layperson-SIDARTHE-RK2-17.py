import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(N, days, t_max, R0, alpha, beta, gamma, delta, epsilon, sigma, theta):
    # Generate time array
    t = np.linspace(0, t_max, days)
    dt = t[1] - t[0]

    # Set initial conditions
    S = np.zeros(days)
    I = np.zeros(days)
    D = np.zeros(days)
    A = np.zeros(days)
    R = np.zeros(days)
    T = np.zeros(days)
    H = np.zeros(days)
    E = np.zeros(days)

    # Set initial values
    S[0] = N - 1
    I[0] = 1

    for i in range(1, days):
        # Update equations
        S[i] = S[i-1] - (R0 * beta * S[i-1] * I[i-1] / N) * dt
        I[i] = I[i-1] + (R0 * beta * S[i-1] * I[i-1] / N - alpha * I[i-1] - gamma * I[i-1]) * dt
        D[i] = D[i-1] + (theta * alpha * I[i-1] - delta * D[i-1]) * dt
        A[i] = A[i-1] + ((1 - theta) * alpha * I[i-1] - epsilon * A[i-1]) * dt
        R[i] = R[i-1] + (gamma * I[i-1] + epsilon * A[i-1]) * dt
        T[i] = D[i] + R[i] + A[i]
        H[i] = T[i] * sigma
        E[i] = (1 - sigma) * T[i]

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Deceased')
    plt.plot(t, A, label='Asymptomatic')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, T, label='Total Cases')
    plt.plot(t, H, label='Hospitalizations')
    plt.plot(t, E, label='Exposures')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.legend()
    plt.title('SIDARTHE Model')
    plt.grid(True)
    plt.show()
}


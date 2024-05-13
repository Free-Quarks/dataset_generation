import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(N, I0, D0, A0, R0, T0, H0, E0, beta, gamma, delta, alpha, theta, k, m, dt, days):
    # Total population, N.
    S0 = N - I0 - D0 - A0 - R0 - T0 - H0 - E0
    # Contact rate, beta, and mean recovery rate, gamma, and mean incubation period, delta, and mean pre-symptomatic period, alpha, and mean hospitalization period, theta, and mean time from hospitalization to death, k
    # A grid of time points (in days)
    t = np.linspace(0, days, int(days/dt))
    # Initialize the SIDARTHE vectors
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    D = np.zeros(len(t))
    A = np.zeros(len(t))
    R = np.zeros(len(t))
    T = np.zeros(len(t))
    H = np.zeros(len(t))
    E = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    D[0] = D0
    A[0] = A0
    R[0] = R0
    T[0] = T0
    H[0] = H0
    E[0] = E0
    # Euler integration
    for i in range(1, len(t)):
        S[i] = S[i-1] - (beta*S[i-1]*(I[i-1] + delta*A[i-1]))/N
        E[i] = E[i-1] + (beta*S[i-1]*(I[i-1] + delta*A[i-1]))/N - alpha*E[i-1]
        I[i] = I[i-1] + alpha*E[i-1] - (theta + gamma)*I[i-1]
        D[i] = D[i-1] + theta*I[i-1] - (k + m)*D[i-1]
        A[i] = A[i-1] + gamma*I[i-1] - (delta + theta)*A[i-1]
        R[i] = R[i-1] + delta*A[i-1] + (1 - m)*D[i-1]
        H[i] = H[i-1] + k*D[i-1]
        T[i] = T[i-1] + m*D[i-1]
    
    return t, S, I, D, A, R, H, E, T


# Example usage
N = 100000
I0, D0, A0, R0, T0, H0, E0 = 10, 0, 0, 0, 0, 0, 0
beta, gamma, delta, alpha, theta, k, m = 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1
dt = 0.1
days = 100

# Run the model
t, S, I, D, A, R, H, E, T = sidarthe_model(N, I0, D0, A0, R0, T0, H0, E0, beta, gamma, delta, alpha, theta, k, m, dt, days)

# Plot the results
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, D, 'g', label='Deceased')
plt.plot(t, A, 'y', label='Asymptomatic')
plt.plot(t, R, 'm', label='Recovered')
plt.plot(t, H, 'c', label='Hospitalized')
plt.plot(t, E, 'k', label='Exposed')
plt.plot(t, T, 'purple', label='Critical')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIDARTHE Model')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt


def serid_rk2(N, beta, gamma, delta, alpha, S0, E0, I0, R0, D0, t_end, dt):
    # Define the differential equations for the model
    def deriv(S, E, I, R, D, t):
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - gamma * E - delta * E
        dIdt = gamma * E - alpha * I
        dRdt = alpha * I
        dDdt = delta * E
        return dSdt, dEdt, dIdt, dRdt, dDdt

    # Initialize arrays to store the values
    S = np.zeros(int(t_end/dt) + 1)
    E = np.zeros(int(t_end/dt) + 1)
    I = np.zeros(int(t_end/dt) + 1)
    R = np.zeros(int(t_end/dt) + 1)
    D = np.zeros(int(t_end/dt) + 1)
    t = np.linspace(0, t_end, int(t_end/dt) + 1)

    # Set initial conditions
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    D[0] = D0

    # Implement the Runge-Kutta 2nd order method
    for i in range(1, int(t_end/dt) + 1):
        k1_S, k1_E, k1_I, k1_R, k1_D = deriv(S[i-1], E[i-1], I[i-1], R[i-1], D[i-1], t[i-1])
        k2_S, k2_E, k2_I, k2_R, k2_D = deriv(S[i-1] + dt/2 * k1_S, E[i-1] + dt/2 * k1_E, I[i-1] + dt/2 * k1_I, R[i-1] + dt/2 * k1_R, D[i-1] + dt/2 * k1_D, t[i-1] + dt/2)
        S[i] = S[i-1] + dt * k2_S
        E[i] = E[i-1] + dt * k2_E
        I[i] = I[i-1] + dt * k2_I
        R[i] = R[i-1] + dt * k2_R
        D[i] = D[i-1] + dt * k2_D

    # Return the arrays
    return S, E, I, R, D


# Example usage
N = 1000
beta = 0.3
gamma = 0.1
alpha = 0.05
delta = 0.05
S0 = N - 10
E0 = 10
I0 = 0
R0 = 0
D0 = 0
t_end = 100
dt = 0.1

S, E, I, R, D = serid_rk2(N, beta, gamma, delta, alpha, S0, E0, I0, R0, D0, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Deaths')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

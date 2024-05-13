import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(beta, gamma, alpha, sigma, rho, delta, dh, di, dr, de, dm, dth, days):
    # Initialize arrays
    S = np.zeros(days)
    I = np.zeros(days)
    D = np.zeros(days)
    A = np.zeros(days)
    R = np.zeros(days)
    T = np.zeros(days)
    H = np.zeros(days)
    E = np.zeros(days)
    M = np.zeros(days)
    Th = np.zeros(days)

    # Set initial conditions
    S[0] = 60000000
    I[0] = 100
    D[0] = 0
    A[0] = 0
    R[0] = 0
    T[0] = 0
    H[0] = 0
    E[0] = 0
    M[0] = 0
    Th[0] = 0

    # Define the differential equations
    def diff_eqn(t, y):
        S, I, D, A, R, T, H, E, M, Th = y
        dS = -beta * S * I / (S + I + D + A + R + T + H + E + M + Th)
        dI = (beta * S * I / (S + I + D + A + R + T + H + E + M + Th)) - (gamma + alpha + sigma + rho + delta) * I
        dD = delta * I
        dA = alpha * I
        dR = gamma * I
        dT = rho * I
        dH = sigma * I
        dE = (1 - delta - alpha - gamma - rho - sigma) * I
        dM = dh * H
        dTh = di * Th
        return [dS, dI, dD, dA, dR, dT, dH, dE, dM, dTh]

    # Time vector
    t = np.linspace(0, days, days)

    # Solve the differential equations using Runge-Kutta 4th order method
    y = np.array([S[0], I[0], D[0], A[0], R[0], T[0], H[0], E[0], M[0], Th[0]])
    for i in range(1, days):
        dt = t[i] - t[i-1]
        k1 = dt * np.array(diff_eqn(t[i-1], y))
        k2 = dt * np.array(diff_eqn(t[i-1] + 0.5*dt, y + 0.5*k1))
        k3 = dt * np.array(diff_eqn(t[i-1] + 0.5*dt, y + 0.5*k2))
        k4 = dt * np.array(diff_eqn(t[i-1] + dt, y + k3))
        y = y + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        S[i], I[i], D[i], A[i], R[i], T[i], H[i], E[i], M[i], Th[i] = y

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Deaths')
    plt.plot(t, A, label='Asymptomatic')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, T, label='ICU')
    plt.plot(t, H, label='Hospitalized')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, M, label='Mild')
    plt.plot(t, Th, label='Therapy')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title('SIDARTHE Model')
    plt.legend()
    plt.grid(True)
    plt.show()
}


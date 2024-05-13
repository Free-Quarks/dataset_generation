import numpy as np
import matplotlib.pyplot as plt


def seirid_model(beta, sigma, gamma1, gamma2, mu, N, I0, E0, R0, D0, t_end, dt):
    # Initialize arrays
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    E = np.zeros(len(t))
    I1 = np.zeros(len(t))
    I2 = np.zeros(len(t))
    R = np.zeros(len(t))
    D = np.zeros(len(t))
    
    # Set initial conditions
    S[0] = N - I0 - E0 - R0 - D0
    E[0] = E0
    I1[0] = I0
    I2[0] = 0
    R[0] = R0
    D[0] = D0
    
    # Define the differential equations
    def diff_eqs(t, y):
        S, E, I1, I2, R, D = y
        dSdt = -beta * S * (I1 + I2) / N + mu * (N - S)
        dEdt = beta * S * (I1 + I2) / N - sigma * E
        dI1dt = sigma * E - gamma1 * I1
        dI2dt = gamma1 * I1 - gamma2 * I2
        dRdt = gamma2 * I2
        dDdt = mu * S
        return [dSdt, dEdt, dI1dt, dI2dt, dRdt, dDdt]
    
    # Solve the differential equations
    for i in range(1, len(t)):
        k1 = dt * np.array(diff_eqs(t[i-1], [S[i-1], E[i-1], I1[i-1], I2[i-1], R[i-1], D[i-1]]))
        k2 = dt * np.array(diff_eqs(t[i-1] + 0.5 * dt, [S[i-1] + 0.5 * k1[0], E[i-1] + 0.5 * k1[1], I1[i-1] + 0.5 * k1[2], I2[i-1] + 0.5 * k1[3], R[i-1] + 0.5 * k1[4], D[i-1] + 0.5 * k1[5]]))
        k3 = dt * np.array(diff_eqs(t[i-1] + dt, [S[i-1] - k1[0] + 2 * k2[0], E[i-1] - k1[1] + 2 * k2[1], I1[i-1] - k1[2] + 2 * k2[2], I2[i-1] - k1[3] + 2 * k2[3], R[i-1] - k1[4] + 2 * k2[4], D[i-1] - k1[5] + 2 * k2[5]]))
        S[i] = S[i-1] + (1/6) * (k1[0] + 4 * k2[0] + k3[0])
        E[i] = E[i-1] + (1/6) * (k1[1] + 4 * k2[1] + k3[1])
        I1[i] = I1[i-1] + (1/6) * (k1[2] + 4 * k2[2] + k3[2])
        I2[i] = I2[i-1] + (1/6) * (k1[3] + 4 * k2[3] + k3[3])
        R[i] = R[i-1] + (1/6) * (k1[4] + 4 * k2[4] + k3[4])
        D[i] = D[i-1] + (1/6) * (k1[5] + 4 * k2[5] + k3[5])
    
    # Plot results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I1, label='Infected 1')
    plt.plot(t, I2, label='Infected 2')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Dead')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

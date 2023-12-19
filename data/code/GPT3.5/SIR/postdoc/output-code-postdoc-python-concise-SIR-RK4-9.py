import numpy as np


def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def RK4_SIR(SIR_model, y0, t, N, beta, gamma):
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0], I[0], R[0] = y0
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        k1 = SIR_model([S[i-1], I[i-1], R[i-1]], t[i-1], N, beta, gamma)
        k2 = SIR_model([S[i-1] + k1[0]*dt/2, I[i-1] + k1[1]*dt/2, R[i-1] + k1[2]*dt/2], t[i-1] + dt/2, N, beta, gamma)
        k3 = SIR_model([S[i-1] + k2[0]*dt/2, I[i-1] + k2[1]*dt/2, R[i-1] + k2[2]*dt/2], t[i-1] + dt/2, N, beta, gamma)
        k4 = SIR_model([S[i-1] + k3[0]*dt, I[i-1] + k3[1]*dt, R[i-1] + k3[2]*dt], t[i-1] + dt, N, beta, gamma)
        S[i] = S[i-1] + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])*dt/6
        I[i] = I[i-1] + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])*dt/6
        R[i] = R[i-1] + (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])*dt/6
    return S, I, R


N = 1000
beta = 0.2
D = 10
gamma = 1/D
S0, I0, R0 = N-1, 1, 0  # initial conditions: one infected, rest susceptible

# Initiate time step
t = np.linspace(0, 99, 100)

# Solve the SIR model using RK4
S, I, R = RK4_SIR(SIR_model, [S0, I0, R0], t, N, beta, gamma)


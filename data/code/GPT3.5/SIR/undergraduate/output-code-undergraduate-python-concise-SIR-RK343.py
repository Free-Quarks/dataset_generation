import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, R0, T):
    dt = 0.1
    t = np.linspace(0, T, int(T/dt)+1)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    S[0] = N - I0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = (beta * S[i-1] * I[i-1] / N) - (gamma * I[i-1])
        dR = gamma * I[i-1]

        S_star = S[i-1] + dS * dt / 3
        I_star = I[i-1] + dI * dt / 3
        R_star = R[i-1] + dR * dt / 3

        dS_star = -beta * S_star * I_star / N
        dI_star = (beta * S_star * I_star / N) - (gamma * I_star)
        dR_star = gamma * I_star

        S[i] = S[i-1] + (dt / 4) * (dS + 3 * dS_star)
        I[i] = I[i-1] + (dt / 4) * (dI + 3 * dI_star)
        R[i] = R[i-1] + (dt / 4) * (dR + 3 * dR_star)

    return t, S, I, R


beta = 0.3
# Infection rate
gamma = 0.1
# Recovery rate
N = 1000
# Total population
I0 = 1
# Initial number of infected
R0 = 0
# Initial number of recovered
T = 100
# Time

# Run the SIR model
t, S, I, R = SIR_RK3(beta, gamma, N, I0, R0, T)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()

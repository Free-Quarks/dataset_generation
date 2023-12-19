import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, t_max):
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.linspace(0, t_max, int(t_max)+1)
    dt = t[1] - t[0]

    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = (beta * S[i-1] * I[i-1] / N) - (gamma * I[i-1])
        dR = gamma * I[i-1]

        S.append(S[i-1] + dS*dt)
        I.append(I[i-1] + dI*dt)
        R.append(R[i-1] + dR*dt)

    return S, I, R


# Example usage
beta = 0.5
gamma = 0.2
N = 1000
I0 = 1
R0 = 0
t_max = 10

S, I, R = SIR_model(beta, gamma, N, I0, R0, t_max)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()

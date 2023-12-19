import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, T):
    dt = 0.01
    t = np.linspace(0, T, int(T/dt)+1)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for i in range(1, len(t)):
        S[i] = S[i-1] - (beta * S[i-1] * I[i-1] / N) * dt
        I[i] = I[i-1] + (beta * S[i-1] * I[i-1] / N - gamma * I[i-1]) * dt
        R[i] = R[i-1] + gamma * I[i-1] * dt
    
    return S, I, R


# Example usage
beta = 0.3
gamma = 0.1
N = 1000
I0 = 1
T = 100

S, I, R = SIR_model(beta, gamma, N, I0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def SEIR_model(beta, sigma, gamma, N, I0, E0, R0, T):
    S0 = N - I0 - E0 - R0
    dt = 0.01
    t = np.arange(0, T, dt)
    S = np.zeros(len(t))
    E = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        S[i] = S[i-1] - beta*S[i-1]*I[i-1]*dt/N
        E[i] = E[i-1] + (beta*S[i-1]*I[i-1] - sigma*E[i-1])*dt
        I[i] = I[i-1] + (sigma*E[i-1] - gamma*I[i-1])*dt
        R[i] = R[i-1] + gamma*I[i-1]*dt

    return S, E, I, R

N = 1000
I0, E0, R0 = 1, 0, 0
beta, sigma, gamma = 0.3, 0.05, 0.1
T = 100

S, E, I, R = SEIR_model(beta, sigma, gamma, N, I0, E0, R0, T)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, N, I0, T):
    dt = 0.1
    t = np.linspace(0, T, int(T/dt)+1)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    for i in range(1, len(t)):
        S[i] = S[i-1] - dt*beta*S[i-1]*I[i-1]/N
        I[i] = I[i-1] + dt*beta*S[i-1]*I[i-1]/N - dt*gamma*I[i-1]
        R[i] = R[i-1] + dt*gamma*I[i-1]
    return S, I, R


beta = 0.5
gamma = 0.1
N = 1000
I0 = 1
T = 10

S, I, R = SIR_RK2(beta, gamma, N, I0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.show()

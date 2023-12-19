import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S0, I0, R0, beta, gamma, t_max, dt):
    t = np.arange(0, t_max, dt)
    N = S0 + I0 + R0
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(len(t)-1):
        S[i+1] = S[i] - (beta*S[i]*I[i]/N)*dt
        I[i+1] = I[i] + (beta*S[i]*I[i]/N - gamma*I[i])*dt
        R[i+1] = R[i] + (gamma*I[i])*dt
    return S, I, R

S0 = 990
I0 = 10
R0 = 0
beta = 0.2
gamma = 0.1
t_max = 100
dt = 0.1

S, I, R = SIR_model(S0, I0, R0, beta, gamma, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.grid(True)
plt.show()

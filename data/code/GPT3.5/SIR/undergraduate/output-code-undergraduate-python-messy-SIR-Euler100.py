import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S0, I0, R0, beta, gamma, t_max, dt):
    t = np.linspace(0, t_max, int(t_max/dt)+1)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        S[i] = S[i-1] - beta*S[i-1]*I[i-1]*dt
        I[i] = I[i-1] + (beta*S[i-1]*I[i-1] - gamma*I[i-1])*dt
        R[i] = R[i-1] + gamma*I[i-1]*dt
    return t, S, I, R


S0 = 0.99
I0 = 0.01
R0 = 0.0
beta = 0.3
gamma = 0.1
t_max = 100
dt = 0.1


t, S, I, R = SIR_model(S0, I0, R0, beta, gamma, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

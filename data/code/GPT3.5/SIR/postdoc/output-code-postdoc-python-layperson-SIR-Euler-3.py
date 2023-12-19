import numpy as np
import matplotlib.pyplot as plt


def SIR_euler_model(beta, gamma, S0, I0, R0, t_max, dt):
    t = np.arange(0, t_max+dt, dt)
    N = S0 + I0 + R0
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        S[i] = S[i-1] - beta*S[i-1]*I[i-1]/N*dt
        I[i] = I[i-1] + (beta*S[i-1]*I[i-1]/N - gamma*I[i-1])*dt
        R[i] = R[i-1] + gamma*I[i-1]*dt
    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_max = 100
dt = 0.1

S, I, R = SIR_euler_model(beta, gamma, S0, I0, R0, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

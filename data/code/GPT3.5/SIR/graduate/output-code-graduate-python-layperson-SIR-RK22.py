import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, S0, I0, R0, t_end, dt):
    t = np.arange(0, t_end, dt)
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
beta = 0.3
gamma = 0.1
S0 = 900
I0 = 100
R0 = 0
t_end = 100
dt = 0.1

S, I, R = SIR_model(beta, gamma, S0, I0, R0, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, t_max, dt):
    
    def S_prime(S, I):
        return -beta * S * I

    def I_prime(S, I):
        return beta * S * I - gamma * I

    def R_prime(I):
        return gamma * I

    N = S0 + I0 + R0
    t = np.arange(0, t_max+dt, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1 = dt * S_prime(S[i-1], I[i-1])
        k2 = dt * S_prime(S[i-1] + k1/2, I[i-1])
        S[i] = S[i-1] + k2

        k1 = dt * I_prime(S[i-1], I[i-1])
        k2 = dt * I_prime(S[i-1], I[i-1] + k1/2)
        I[i] = I[i-1] + k2

        k1 = dt * R_prime(I[i-1])
        k2 = dt * R_prime(I[i-1] + k1/2)
        R[i] = R[i-1] + k2

    return S, I, R

beta = 0.3
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0.0
t_max = 100

S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, t_max, dt=0.1)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()

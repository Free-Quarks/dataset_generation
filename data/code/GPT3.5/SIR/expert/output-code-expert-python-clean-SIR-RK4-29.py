import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, S0, I0, R0, T, N):
    def derivs(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    y0 = S0, I0, R0
    t = np.linspace(0, T, T+1)
    
    res = odeint(derivs, y0, t, args=(N, beta, gamma))
    S, I, R = res.T
    
    return S, I, R

# Example usage:
beta = 0.2
gamma = 0.1
S0 = 999
I0 = 1
R0 = 0
T = 100
N = S0 + I0 + R0
S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, T, N)

t = np.linspace(0, T, T+1)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

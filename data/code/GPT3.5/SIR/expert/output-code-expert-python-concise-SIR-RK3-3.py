import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, days):
    def derivs(SIR, t):
        S, I, R = SIR
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    S0 = N - I0
    R0 = 0
    t = np.linspace(0, days, days)
    SIR0 = S0, I0, R0
    odeint(derivs, SIR0, t)
    
    plt.plot(t, SIR[:, 0], label='S')
    plt.plot(t, SIR[:, 1], label='I')
    plt.plot(t, SIR[:, 2], label='R')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model using RK3')
    plt.legend()
    plt.show()

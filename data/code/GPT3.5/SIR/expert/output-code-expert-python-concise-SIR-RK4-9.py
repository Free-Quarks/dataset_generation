import numpy as np
import matplotlib.pyplot as plt


def SIR_model(S0, I0, R0, beta, gamma, t)
    def deriv(SIR, t):
        S,I,R = SIR
        dS_dt = -beta*S*I
        dI_dt = beta*S*I - gamma*I
        dR_dt = gamma*I
        return [dS_dt, dI_dt, dR_dt]
    
    SIR0 = [S0, I0, R0]
    solution = odeint(deriv, SIR0, t)
    
    S = solution[:, 0]
    I = solution[:, 1]
    R = solution[:, 2]
    
    plt.plot(t, S, label='S(t)')
    plt.plot(t, I, label='I(t)')
    plt.plot(t, R, label='R(t)')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()

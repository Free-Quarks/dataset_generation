import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, N, I0, T):
    def dSdt(S, I):
        return -beta * S * I / N
    
    def dIdt(S, I):
        return beta * S * I / N - gamma * I
    
    def dRdt(I):
        return gamma * I
    
    def RK2_step(S, I, R, dt):
        S1 = S + dt * dSdt(S, I)
        I1 = I + dt * dIdt(S, I)
        R1 = R + dt * dRdt(I)
        S2 = S + dt * dSdt(S1, I1)
        I2 = I + dt * dIdt(S1, I1)
        R2 = R + dt * dRdt(I1)
        S_new = (S1 + S2) / 2
        I_new = (I1 + I2) / 2
        R_new = (R1 + R2) / 2
        return S_new, I_new, R_new
    
    S = N - I0
    I = I0
    R = 0
    t = 0
    dt = 0.1
    
    S_values = [S]
    I_values = [I]
    R_values = [R]
    t_values = [t]
    
    while t < T:
        S, I, R = RK2_step(S, I, R, dt)
        t += dt
        S_values.append(S)
        I_values.append(I)
        R_values.append(R)
        t_values.append(t)
    
    plt.plot(t_values, S_values, label='Susceptible')
    plt.plot(t_values, I_values, label='Infected')
    plt.plot(t_values, R_values, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()
}


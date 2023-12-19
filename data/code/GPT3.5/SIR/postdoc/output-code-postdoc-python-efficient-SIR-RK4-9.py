import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(beta, gamma, S0, I0, R0, t_end, dt):
    def SIR_model(S, I, R, beta, gamma):
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = [0]
    
    while t[-1] < t_end:
        S_last, I_last, R_last = S[-1], I[-1], R[-1]
        k1_S, k1_I, k1_R = SIR_model(S_last, I_last, R_last, beta, gamma)
        k2_S, k2_I, k2_R = SIR_model(S_last + 0.5 * dt * k1_S, I_last + 0.5 * dt * k1_I, R_last + 0.5 * dt * k1_R, beta, gamma)
        k3_S, k3_I, k3_R = SIR_model(S_last + 0.5 * dt * k2_S, I_last + 0.5 * dt * k2_I, R_last + 0.5 * dt * k2_R, beta, gamma)
        k4_S, k4_I, k4_R = SIR_model(S_last + dt * k3_S, I_last + dt * k3_I, R_last + dt * k3_R, beta, gamma)
        
        S_new = S_last + (dt / 6.0) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        I_new = I_last + (dt / 6.0) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)
        R_new = R_last + (dt / 6.0) * (k1_R + 2 * k2_R + 2 * k3_R + k4_R)
        t_new = t[-1] + dt
        
        S.append(S_new)
        I.append(I_new)
        R.append(R_new)
        t.append(t_new)
        
    return S, I, R, t


beta = 0.3
gamma = 0.1
S0 = 99
I0 = 1
R0 = 0
t_end = 100
dt = 0.1

S, I, R, t = SIR_RK4(beta, gamma, S0, I0, R0, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model Simulation')
plt.show()

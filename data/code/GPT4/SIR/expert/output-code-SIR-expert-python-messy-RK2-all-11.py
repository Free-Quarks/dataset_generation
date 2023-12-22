import numpy as np
import matplotlib.pyplot as plt

def SIR_model(t, Y, beta, gamma):
    S, I, R = Y
    dS_dt = - beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]

def RK2(Y, h, t, beta, gamma):
    k1 = h * np.array(SIR_model(t, Y, beta, gamma))
    k2 = h * np.array(SIR_model(t + 0.5 * h, Y + 0.5 * k1, beta, gamma))
    Y = Y + k2
    return Y

def simulate_SIR(S0, I0, R0, beta, gamma, T, dt):
    N_t = int(T / dt)  
    t = np.linspace(0, N_t*dt, N_t+1)  
    S = np.zeros(N_t+1)
    I = np.zeros(N_t+1)
    R = np.zeros(N_t+1) 
    S[0] = S0
    I[0] = I0
    R[0] = R0
    Y = [S0, I0, R0]  
    for i in range(N_t):
        Y = RK2(Y, dt, i*dt, beta, gamma)
        S[i+1], I[i+1], R[i+1] = Y
    return t, S, I, R

t, S, I, R = simulate_SIR(999, 1, 0, 0.5, 0.1, 150, 0.5)

plt.plot(t, S, label='S(t)')
plt.plot(t, I, label='I(t)')
plt.plot(t, R, label='R(t)')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Number')
plt.show()

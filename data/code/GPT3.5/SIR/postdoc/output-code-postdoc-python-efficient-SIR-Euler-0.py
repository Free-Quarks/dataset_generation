import numpy as np
import matplotlib.pyplot as plt


def sir_model(S, I, R, beta, gamma, N, dt):
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    
    S_new = S + dSdt * dt
    I_new = I + dIdt * dt
    R_new = R + dRdt * dt
    
    return S_new, I_new, R_new


def simulate_sir_model(S0, I0, R0, beta, gamma, N, dt, T):
    n_steps = int(T / dt)
    
    S = np.zeros(n_steps)
    I = np.zeros(n_steps)
    R = np.zeros(n_steps)
    
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, n_steps):
        S[i], I[i], R[i] = sir_model(S[i-1], I[i-1], R[i-1], beta, gamma, N, dt)
    
    t = np.linspace(0, T, n_steps)
    
    return S, I, R, t


# Example usage
S0 = 800
I0 = 200
R0 = 0
beta = 0.5
gamma = 0.1
N = 1000
dt = 0.1
T = 100

S, I, R, t = simulate_sir_model(S0, I0, R0, beta, gamma, N, dt, T)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.show()

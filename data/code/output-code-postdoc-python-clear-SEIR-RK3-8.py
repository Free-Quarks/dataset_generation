import numpy as np
import matplotlib.pyplot as plt

def SEIR_RK3(beta, gamma, sigma, N, I0, E0, R0, t_end, delta_t):
    S0 = N - I0 - E0 - R0
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    t = [0]
    
    while t[-1] < t_end:
        t_current = t[-1]
        S_current = S[-1]
        E_current = E[-1]
        I_current = I[-1]
        R_current = R[-1]
        
        dS = -beta*S_current*I_current/N
        dE = beta*S_current*I_current/N - sigma*E_current
        dI = sigma*E_current - gamma*I_current
        dR = gamma*I_current
        
        S_next = S_current + delta_t*dS
        E_next = E_current + delta_t*dE
        I_next = I_current + delta_t*dI
        R_next = R_current + delta_t*dR
        t_next = t_current + delta_t
        
        S.append(S_next)
        E.append(E_next)
        I.append(I_next)
        R.append(R_next)
        t.append(t_next)
    
    return S, E, I, R, t


# Example usage:
beta = 0.3
gamma = 0.1
sigma = 0.2
N = 1000
I0 = 1
E0 = 0
R0 = 0
t_end = 50
delta_t = 0.1

S, E, I, R, t = SEIR_RK3(beta, gamma, sigma, N, I0, E0, R0, t_end, delta_t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()

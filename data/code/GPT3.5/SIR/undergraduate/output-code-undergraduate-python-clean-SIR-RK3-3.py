import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, S0, I0, R0, t_total, dt):
    N = S0 + I0 + R0
    S = np.zeros(t_total)
    I = np.zeros(t_total)
    R = np.zeros(t_total)
    t = np.linspace(0, t_total, t_total)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(t_total-1):
        S_prime = -beta*S[i]*I[i]/N
        I_prime = beta*S[i]*I[i]/N - gamma*I[i]
        R_prime = gamma*I[i]
        
        S_star = S[i] + dt*S_prime
        I_star = I[i] + dt*I_prime
        R_star = R[i] + dt*R_prime
        
        S_prime_star = -beta*S_star*I_star/N
        I_prime_star = beta*S_star*I_star/N - gamma*I_star
        R_prime_star = gamma*I_star
        
        S_new = (3/4)*S[i] + (1/4)*S_star + (1/4)*dt*S_prime_star
        I_new = (3/4)*I[i] + (1/4)*I_star + (1/4)*dt*I_prime_star
        R_new = (3/4)*R[i] + (1/4)*R_star + (1/4)*dt*R_prime_star
        
        S[i+1] = S_new
        I[i+1] = I_new
        R[i+1] = R_new
    
    return S, I, R

beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_total = 100
dt = 0.1

S, I, R = SIR_RK3(beta, gamma, S0, I0, R0, t_total, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, T):
    dt = 0.1
    t = np.linspace(0, T, int(T/dt)+1)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for i in range(len(t)-1):
        dSdt = -beta*S[i]*I[i]/N
        dIdt = beta*S[i]*I[i]/N - gamma*I[i]
        dRdt = gamma*I[i]
        
        k1_S = dt*dSdt
        k1_I = dt*dIdt
        k1_R = dt*dRdt
        
        k2_S = dt*(-beta*(S[i]+0.5*k1_S)*(I[i]+0.5*k1_I)/N)
        k2_I = dt*(beta*(S[i]+0.5*k1_S)*(I[i]+0.5*k1_I)/N - gamma*(I[i]+0.5*k1_I))
        k2_R = dt*gamma*(I[i]+0.5*k1_I)
        
        k3_S = dt*(-beta*(S[i]-k1_S+2*k2_S)*(I[i]-k1_I+2*k2_I)/N)
        k3_I = dt*(beta*(S[i]-k1_S+2*k2_S)*(I[i]-k1_I+2*k2_I)/N - gamma*(I[i]-k1_I+2*k2_I))
        k3_R = dt*gamma*(I[i]-k1_I+2*k2_I)
        
        S[i+1] = S[i] + (1/6)*(k1_S + 4*k2_S + k3_S)
        I[i+1] = I[i] + (1/6)*(k1_I + 4*k2_I + k3_I)
        R[i+1] = R[i] + (1/6)*(k1_R + 4*k2_R + k3_R)
        
    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
T = 100

S, I, R = SIR_model(beta, gamma, N, I0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

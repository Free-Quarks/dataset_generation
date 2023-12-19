import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, days):
    def derivative(y, t, beta, gamma, N):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    y0 = N - I0, I0, 0
    t = np.linspace(0, days, days+1)
    h = t[1] - t[0]
    
    S = np.zeros(days+1)
    I = np.zeros(days+1)
    R = np.zeros(days+1)
    
    S[0], I[0], R[0] = y0
    
    for i in range(1, days+1):
        k1 = derivative([S[i-1], I[i-1], R[i-1]], t[i-1], beta, gamma, N)
        k2 = derivative([S[i-1] + h * k1[0]/2, I[i-1] + h * k1[1]/2, R[i-1] + h * k1[2]/2], t[i-1] + h/2, beta, gamma, N)
        k3 = derivative([S[i-1] - h * k1[0] + 2 * h * k2[0], I[i-1] - h * k1[1] + 2 * h * k2[1], R[i-1] - h * k1[2] + 2 * h * k2[2]], t[i-1] + h, beta, gamma, N)
        
        S[i] = S[i-1] + h * (k1[0] + 4 * k2[0] + k3[0]) / 6
        I[i] = I[i-1] + h * (k1[1] + 4 * k2[1] + k3[1]) / 6
        R[i] = R[i-1] + h * (k1[2] + 4 * k2[2] + k3[2]) / 6
    
    return S, I, R

# Example usage
beta = 0.4
gamma = 0.1
N = 1000
I0 = 1
days = 100

S, I, R = SIR_RK3(beta, gamma, N, I0, days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.title('SIR model using RK3')
plt.show()

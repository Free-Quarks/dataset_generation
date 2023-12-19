import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(S0, I0, R0, beta, gamma, t_end, N, h):
    # Initializing arrays
    S = np.zeros(t_end)
    I = np.zeros(t_end)
    R = np.zeros(t_end)
    t = np.linspace(0, t_end, t_end)
    
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta 4th order method
    for i in range(1, t_end):
        k1 = h * (-beta * S[i-1] * I[i-1] / N)
        l1 = h * (beta * S[i-1] * I[i-1] / N - gamma * I[i-1])
        m1 = h * (gamma * I[i-1])
        
        k2 = h * (-beta * (S[i-1] + 0.5 * k1) * (I[i-1] + 0.5 * l1) / N)
        l2 = h * (beta * (S[i-1] + 0.5 * k1) * (I[i-1] + 0.5 * l1) / N - gamma * (I[i-1] + 0.5 * l1))
        m2 = h * (gamma * (I[i-1] + 0.5 * l1))
        
        k3 = h * (-beta * (S[i-1] + 0.5 * k2) * (I[i-1] + 0.5 * l2) / N)
        l3 = h * (beta * (S[i-1] + 0.5 * k2) * (I[i-1] + 0.5 * l2) / N - gamma * (I[i-1] + 0.5 * l2))
        m3 = h * (gamma * (I[i-1] + 0.5 * l2))
        
        k4 = h * (-beta * (S[i-1] + k3) * (I[i-1] + l3) / N)
        l4 = h * (beta * (S[i-1] + k3) * (I[i-1] + l3) / N - gamma * (I[i-1] + l3))
        m4 = h * (gamma * (I[i-1] + l3))
        
        S[i] = S[i-1] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        I[i] = I[i-1] + (1/6) * (l1 + 2*l2 + 2*l3 + l4)
        R[i] = R[i-1] + (1/6) * (m1 + 2*m2 + 2*m3 + m4)
    
    return S, I, R


# Example usage
S0 = 999
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
t_end = 100
N = S0 + I0 + R0
h = 1

S, I, R = SIR_RK4(S0, I0, R0, beta, gamma, t_end, N, h)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK4')
plt.show()

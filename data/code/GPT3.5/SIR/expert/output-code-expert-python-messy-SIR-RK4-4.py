import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt):
    N = S0 + I0 + R0
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    t = np.arange(0, t_max, dt)
    
    for i in range(1, t_max):
        k1 = dt * (-beta * S[i-1] * I[i-1] / N)
        l1 = dt * (beta * S[i-1] * I[i-1] / N - gamma * I[i-1])
        m1 = dt * (gamma * I[i-1])
        
        k2 = dt * (-beta * (S[i-1] + 0.5 * k1) * (I[i-1] + 0.5 * l1) / N)
        l2 = dt * (beta * (S[i-1] + 0.5 * k1) * (I[i-1] + 0.5 * l1) / N - gamma * (I[i-1] + 0.5 * l1))
        m2 = dt * (gamma * (I[i-1] + 0.5 * l1))
        
        k3 = dt * (-beta * (S[i-1] + 0.5 * k2) * (I[i-1] + 0.5 * l2) / N)
        l3 = dt * (beta * (S[i-1] + 0.5 * k2) * (I[i-1] + 0.5 * l2) / N - gamma * (I[i-1] + 0.5 * l2))
        m3 = dt * (gamma * (I[i-1] + 0.5 * l2))
        
        k4 = dt * (-beta * (S[i-1] + k3) * (I[i-1] + l3) / N)
        l4 = dt * (beta * (S[i-1] + k3) * (I[i-1] + l3) / N - gamma * (I[i-1] + l3))
        m4 = dt * (gamma * (I[i-1] + l3))
        
        S[i] = S[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        I[i] = I[i-1] + (l1 + 2 * l2 + 2 * l3 + l4) / 6
        R[i] = R[i-1] + (m1 + 2 * m2 + 2 * m3 + m4) / 6
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()
}


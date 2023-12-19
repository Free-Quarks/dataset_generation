import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(N, beta, gamma, delta_t, num_steps):
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    t = np.zeros(num_steps)
    
    S[0] = N - 1
    I[0] = 1
    R[0] = 0
    t[0] = 0
    
    for i in range(1, num_steps):
        k1_S = -beta * S[i-1] * I[i-1] / N
        k1_I = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k1_R = gamma * I[i-1]
        
        S_half = S[i-1] + k1_S * delta_t / 2
        I_half = I[i-1] + k1_I * delta_t / 2
        R_half = R[i-1] + k1_R * delta_t / 2
        
        k2_S = -beta * S_half * I_half / N
        k2_I = beta * S_half * I_half / N - gamma * I_half
        k2_R = gamma * I_half
        
        S_new = S[i-1] + k2_S * delta_t
        I_new = I[i-1] + k2_I * delta_t
        R_new = R[i-1] + k2_R * delta_t
        
        k3_S = -beta * S_new * I_new / N
        k3_I = beta * S_new * I_new / N - gamma * I_new
        k3_R = gamma * I_new
        
        S[i] = S[i-1] + (k1_S + 4*k2_S + k3_S) * delta_t / 6
        I[i] = I[i-1] + (k1_I + 4*k2_I + k3_I) * delta_t / 6
        R[i] = R[i-1] + (k1_R + 4*k2_R + k3_R) * delta_t / 6
        t[i] = t[i-1] + delta_t
    
    return t, S, I, R


N = 1000
beta = 0.3
gamma = 0.1
num_steps = 100
delta_t = 0.1


def main():
    t, S, I, R = SIR_RK3(N, beta, gamma, delta_t, num_steps)
    
    plt.plot(t, S, label='S')
    plt.plot(t, I, label='I')
    plt.plot(t, R, label='R')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

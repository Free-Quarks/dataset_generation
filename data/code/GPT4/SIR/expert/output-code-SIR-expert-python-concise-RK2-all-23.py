import numpy as np
import matplotlib.pyplot as plt
import json

def rk2_sir_model(N, beta, gamma, I0, R0, T):

    S0 = N - I0 - R0
    dt = 1.0
    Nt = int(T/dt)

    S = np.zeros(Nt+1)
    I = np.zeros(Nt+1)
    R = np.zeros(Nt+1)

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for t in range(Nt):
        St = S[t]
        It = I[t]
        Rt = R[t]
        
        k1S = -beta*St*It/N
        k1I = beta*St*It/N - gamma*It
        k1R = gamma*It
        
        k2S = -beta*(St+0.5*dt*k1S)*(It+0.5*dt*k1I)/N
        k2I = beta*(St+0.5*dt*k1S)*(It+0.5*dt*k1I)/N - gamma*(It+0.5*dt*k1I)
        k2R = gamma*(It+0.5*dt*k1I)
        
        S[t+1] = St + dt*k2S
        I[t+1] = It + dt*k2I
        R[t+1] = Rt + dt*k2R

    plt.figure()
    plt.plot(S, 'b', label='Susceptible')
    plt.plot(I, 'r', label='Infected')
    plt.plot(R, 'g', label='Recovered')
    plt.legend()
    plt.show()

rk2_sir_model(500, 0.2, 0.1, 1, 0, 100)

import numpy as np
import matplotlib.pyplot as plt
import json

def RK3_SIR(y, N, beta, gamma, dt, t_end):
    S, I, R = y
    t = 0
    S_list, I_list, R_list, t_list = [S], [I], [R], [t]
    
    while t < t_end:
        k1_S = -beta * S * I / N
        k1_I = beta * S * I / N - gamma * I
        k1_R = gamma * I

        k2_S = -(beta * (S + dt/2 * k1_S) * (I + dt/2 * k1_I)) / N
        k2_I = (beta * (S + dt/2 * k1_S) * (I + dt/2 * k1_I)) / N - gamma * (I + dt/2 * k1_I)
        k2_R = gamma * (I + dt/2 * k1_I)

        k3_S = -(beta * (S + dt * k2_S) * (I + dt * k2_I)) / N
        k3_I = (beta * (S + dt * k2_S) * (I + dt * k2_I)) / N - gamma * (I + dt * k2_I)
        k3_R = gamma * (I + dt * k2_I)

        S += dt/6 * (k1_S + 4*k2_S + k3_S)
        I += dt/6 * (k1_I + 4*k2_I + k3_I)
        R += dt/6 * (k1_R + 4*k2_R + k3_R)

        t += dt

        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
        t_list.append(t)

    plt.figure(figsize=[6,4])
    plt.plot(t_list, S_list, label='Susceptible')
    plt.plot(t_list, I_list, label='Infected')
    plt.plot(t_list, R_list, label='Recovered')
    plt.legend()
    plt.grid()
    plt.show()

N = 1000
I0 = 1
S0 = N - I0
R0 = 0
y0 = S0, I0, R0

beta = 0.3
gamma = 0.1

RK3_SIR(y0, N, beta, gamma, 0.1, 100)

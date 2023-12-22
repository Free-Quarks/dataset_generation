import numpy as np
import matplotlib.pyplot as plt
import json

def rk2_sir(y, h, N, beta, gamma):
    S, I, R = y
    Sn = S - h * 0.5 * beta * S * I / N
    In = I + h * 0.5 * (beta * S * I / N - gamma * I)
    Rn = R + h * 0.5 * gamma * I
    S = S - h * beta * Sn * In / N
    I = I + h * (beta * Sn * In / N - gamma * In)
    R = R + h * gamma * In
    return S, I, R

def main():
    N = 1000
    I0 = 1
    S0 = N - I0
    R0 = 0.0
    beta = 0.2
    gamma = 0.1

    numDays = 150
    dt = 1.0

    S, I, R = [S0], [I0], [R0]
    for time in range(numDays):
        next_S, next_I, next_R = rk2_sir([S[-1], I[-1], R[-1]], dt, N, beta, gamma)
        S.append(next_S)
        I.append(next_I)
        R.append(next_R)
        
    days = np.linspace(0, numDays, numDays+1)
    plt.figure(figsize=[6,4])
    plt.plot(days, S, 'b', label='S')
    plt.plot(days, I, 'r', label='I')
    plt.plot(days, R, 'g', label='R')
    plt.legend()
    plt.grid()
    plt.xlabel('Time /days')
    plt.ylabel('Number')
    plt.title('SIR model with RK2')
    plt.show()

if __name__ == "__main__":
    main()

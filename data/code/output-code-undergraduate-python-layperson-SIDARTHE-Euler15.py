import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(R0: float, t_inc: float, t_rec: float, t_hos: float, t_icu: float, t_death: float, t_imm: float, pop_size: int, t_max: int):
    # Initialize arrays
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    D = np.zeros(t_max)
    A = np.zeros(t_max)
    R = np.zeros(t_max)
    T = np.zeros(t_max)
    H = np.zeros(t_max)
    IC = np.zeros(t_max)
    E = np.zeros(t_max)

    # Initial values
    E[0] = 1
    I[0] = 1

    # Euler's method
    dt = 0.1
    for t in range(1, t_max):
        S[t] = S[t-1] - dt * R0 / t_inc * S[t-1] * I[t-1] / pop_size
        E[t] = E[t-1] + dt * R0 / t_inc * S[t-1] * I[t-1] / pop_size - dt / t_inc * E[t-1]
        I[t] = I[t-1] + dt / t_inc * E[t-1] - dt / (t_rec - t_hos - t_icu - t_death) * I[t-1]
        D[t] = D[t-1] + dt / (t_rec - t_hos - t_icu - t_death) * I[t-1] - dt / t_death * D[t-1]
        A[t] = A[t-1] + dt / (t_rec - t_hos - t_icu - t_death) * I[t-1]
        H[t] = H[t-1] + dt / t_hos * I[t-1] - dt / (t_rec - t_hos) * H[t-1]
        IC[t] = IC[t-1] + dt / t_icu * H[t-1] - dt / t_hos * IC[t-1]
        R[t] = R[t-1] + dt / (t_rec - t_hos - t_icu - t_death) * (I[t-1] + H[t-1])
        T[t] = T[t-1] + dt / (t_rec - t_hos - t_icu - t_death) * (I[t-1] + H[t-1] + R[t-1] + D[t-1])
    
    return S, E, I, D, A, R, T, H, IC


def plot_sidarthe(S, E, I, D, A, R, T, H, IC):
    t = np.arange(len(S))

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Dead')
    plt.plot(t, A, label='Asymptomatic')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, T, label='Total cases')
    plt.plot(t, H, label='Hospitalized')
    plt.plot(t, IC, label='ICU')
    
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.legend()
    plt.show()

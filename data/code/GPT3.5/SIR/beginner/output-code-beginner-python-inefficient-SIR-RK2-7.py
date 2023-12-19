import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(S, I, R, beta, gamma, t_max, dt):
    N = S + I + R
    t = np.linspace(0, t_max, int(t_max/dt)+1)
    S_values = [S]
    I_values = [I]
    R_values = [R]

    for i in range(1, len(t)):
        k1_S = -beta * S * I / N
        k1_I = beta * S * I / N - gamma * I
        k1_R = gamma * I

        S_half = S + 0.5 * dt * k1_S
        I_half = I + 0.5 * dt * k1_I
        R_half = R + 0.5 * dt * k1_R

        k2_S = -beta * S_half * I_half / N
        k2_I = beta * S_half * I_half / N - gamma * I_half
        k2_R = gamma * I_half

        S += dt * k2_S
        I += dt * k2_I
        R += dt * k2_R

        S_values.append(S)
        I_values.append(I)
        R_values.append(R)

    return t, S_values, I_values, R_values


def plot_SIR(t, S, I, R):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Example usage
S = 990
I = 10
R = 0
beta = 0.3
gamma = 0.1

t, S_values, I_values, R_values = SIR_RK2(S, I, R, beta, gamma, t_max=100, dt=0.1)
plot_SIR(t, S_values, I_values, R_values)

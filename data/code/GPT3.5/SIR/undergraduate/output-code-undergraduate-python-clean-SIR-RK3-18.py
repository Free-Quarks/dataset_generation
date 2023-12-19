import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, N, I0, T):
    # Set initial conditions
    S0 = N - I0
    R0 = 0
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.linspace(0, T, num=1000)
    dt = t[1] - t[0]

    # Define the derivatives
    def dSdt(S, I):
        return -beta * S * I / N

    def dIdt(S, I):
        return beta * S * I / N - gamma * I

    def dRdt(I):
        return gamma * I

    # Runge-Kutta method
    for i in range(len(t)-1):
        k1_S = dSdt(S[i], I[i])
        k1_I = dIdt(S[i], I[i])
        k1_R = dRdt(I[i])

        k2_S = dSdt(S[i] + 0.5 * dt * k1_S, I[i] + 0.5 * dt * k1_I)
        k2_I = dIdt(S[i] + 0.5 * dt * k1_S, I[i] + 0.5 * dt * k1_I)
        k2_R = dRdt(I[i] + 0.5 * dt * k1_I)

        k3_S = dSdt(S[i] - dt * k1_S + 2 * dt * k2_S, I[i] - dt * k1_I + 2 * dt * k2_I)
        k3_I = dIdt(S[i] - dt * k1_S + 2 * dt * k2_S, I[i] - dt * k1_I + 2 * dt * k2_I)
        k3_R = dRdt(I[i] - dt * k1_I + 2 * dt * k2_I)

        S_next = S[i] + (dt / 6) * (k1_S + 4 * k2_S + k3_S)
        I_next = I[i] + (dt / 6) * (k1_I + 4 * k2_I + k3_I)
        R_next = R[i] + (dt / 6) * (k1_R + 4 * k2_R + k3_R)

        S.append(S_next)
        I.append(I_next)
        R.append(R_next)

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model using RK3')
    plt.legend()
    plt.show()


# Example usage
def main():
    beta = 0.2
    gamma = 0.1
    N = 1000
    I0 = 10
    T = 100
    SIR_RK3(beta, gamma, N, I0, T)


if __name__ == '__main__':
    main()

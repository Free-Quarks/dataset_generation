import numpy as np
import matplotlib.pyplot as plt


def sir_rk2(beta, gamma, N, I0, R0, t_end, dt):
    # Initial conditions
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.arange(0, t_end, dt)

    # Runge-Kutta 2nd order integration
    for i in range(1, len(t)):
        # Calculate derivatives
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]

        # Calculate intermediate values
        S_star = S[i-1] + dt * dSdt
        I_star = I[i-1] + dt * dIdt

        # Calculate final values
        S.append(S[i-1] + dt * (dSdt + (-beta * S_star * I_star / N)))
        I.append(I[i-1] + dt * (dIdt + (beta * S_star * I_star / N - gamma * I_star)))
        R.append(R[i-1] + dt * (dRdt + gamma * I_star))

    return S, I, R


# Example usage
beta = 0.4
gamma = 0.2
N = 1000
I0 = 1
R0 = 0
t_end = 100
dt = 0.1

S, I, R = sir_rk2(beta, gamma, N, I0, R0, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()

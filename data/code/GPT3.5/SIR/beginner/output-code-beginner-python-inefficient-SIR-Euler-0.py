import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, S0, I0, R0, T):
    N = S0 + I0 + R0
    S, I, R = [S0], [I0], [R0]
    dt = T[1] - T[0]
    for _ in T[1:]:
        S_to_I = (beta * S[-1] * I[-1] / N) * dt
        I_to_R = (gamma * I[-1]) * dt
        S.append(S[-1] - S_to_I)
        I.append(I[-1] + S_to_I - I_to_R)
        R.append(R[-1] + I_to_R)
    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
T = np.linspace(0, 100, 1000)

S, I, R = SIR_model(beta, gamma, S0, I0, R0, T)

plt.plot(T, S, label='Susceptible')
plt.plot(T, I, label='Infected')
plt.plot(T, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

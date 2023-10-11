import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(N, I0, D0, A0, R0, T0, H0, E0, days, alpha, beta, gamma, delta, theta, mu):
    S0 = N - I0 - D0 - A0 - R0 - T0 - H0 - E0
    S = [S0]
    I = [I0]
    D = [D0]
    A = [A0]
    R = [R0]
    T = [T0]
    H = [H0]
    E = [E0]
    for day in range(days):
        S2I = min(beta * I[-1] * S[-1] / N, S[-1])
        I2D = min(delta * I[-1], I[-1])
        I2A = min(theta * I[-1], I[-1])
        A2R = min(gamma * A[-1], A[-1])
        I2T = min(mu * I[-1], I[-1])
        I2H = min(alpha * I[-1], I[-1])
        S.append(S[-1] - S2I)
        I.append(I[-1] + S2I - I2D - I2A - I2T - I2H)
        D.append(D[-1] + I2D)
        A.append(A[-1] + I2A - A2R)
        R.append(R[-1] + A2R + I2T)
        T.append(T[-1] + I2T)
        H.append(H[-1] + I2H)
        E.append(S2I)
    return S, I, D, A, R, T, H, E


N = 100000
I0 = 1
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0

# Set the parameters
beta = 0.5
gamma = 0.1
theta = 0.05
mu = 0.05
alpha = 0.02
delta = 0.01

# Set the number of days
days = 100

# Run the model
S, I, D, A, R, T, H, E = sidarthe_model(N, I0, D0, A0, R0, T0, H0, E0, days, alpha, beta, gamma, delta, theta, mu)

# Plot the results
plt.plot(range(days+1), S, label='Susceptible')
plt.plot(range(days+1), I, label='Infected')
plt.plot(range(days+1), D, label='Deceased')
plt.plot(range(days+1), A, label='Asymptomatic')
plt.plot(range(days+1), R, label='Recovered')
plt.plot(range(days+1), T, label='Tested')
plt.plot(range(days+1), H, label='Hospitalized')
plt.plot(range(days+1), E, label='Exposed')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.show()

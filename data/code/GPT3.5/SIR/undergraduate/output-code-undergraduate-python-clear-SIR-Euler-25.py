import matplotlib.pyplot as plt


def SIR_model(beta, gamma, S0, I0, R0, T):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]

    dt = 1

    for _ in range(T):
        dS = -beta * S[-1] * I[-1] / N
        dI = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dR = gamma * I[-1]

        S.append(S[-1] + dt * dS)
        I.append(I[-1] + dt * dI)
        R.append(R[-1] + dt * dR)

    return S, I, R


beta = 0.2
gamma = 0.1
S0 = 900
I0 = 100
R0 = 0
T = 100

S, I, R = SIR_model(beta, gamma, S0, I0, R0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()


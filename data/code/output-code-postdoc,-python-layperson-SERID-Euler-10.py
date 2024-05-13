import matplotlib.pyplot as plt


def serid_model(beta, gamma, N, I0, R0, D0, t_max):
    S0 = N - I0 - R0 - D0
    S = [S0]
    E = [0]
    I = [I0]
    R = [R0]
    D = [D0]
    dt = 0.1
    t = [0]
    while t[-1] < t_max:
        S.append(S[-1] - beta * I[-1] * S[-1] * dt)
        E.append(E[-1] + beta * I[-1] * S[-1] * dt - gamma * E[-1] * dt)
        I.append(I[-1] + gamma * E[-1] * dt)
        R.append(R[-1] + gamma * I[-1] * dt)
        D.append(D[-1] + gamma * I[-1] * dt)
        t.append(t[-1] + dt)
    return S, E, I, R, D


N = 1000
I0 = 1
R0 = 0
D0 = 0
beta = 0.2
gamma = 0.1
t_max = 100

S, E, I, R, D = serid_model(beta, gamma, N, I0, R0, D0, t_max)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SERID Model')
plt.legend()
plt.show()

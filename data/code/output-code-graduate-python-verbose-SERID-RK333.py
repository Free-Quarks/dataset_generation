import numpy as np
import matplotlib.pyplot as plt


def serid_rk3(beta, gamma, p, q, r, d, S0, E0, I0, R0, D0, tf, h):
    N = S0 + E0 + I0 + R0 + D0
    S, E, I, R, D = [S0], [E0], [I0], [R0], [D0]
    t = np.arange(0, tf + h, h)

    for ts in t[1:]:
        Ss, Es, Is, Rs, Ds = S[-1], E[-1], I[-1], R[-1], D[-1]

        k1 = -(beta * Is * Ss) / N
        l1 = (beta * Is * Ss) / N - p * Es
        m1 = p * Es - q * Is
        n1 = q * Is - gamma * Rs
        o1 = gamma * Rs + r * Is + d * Is

        k2 = -(beta * (Is + h / 2 * m1) * (Ss + h / 2 * k1)) / N
        l2 = (beta * (Is + h / 2 * m1) * (Ss + h / 2 * k1)) / N - p * (Es + h / 2 * l1)
        m2 = p * (Es + h / 2 * l1) - q * (Is + h / 2 * m1)
        n2 = q * (Is + h / 2 * m1) - gamma * (Rs + h / 2 * n1)
        o2 = gamma * (Rs + h / 2 * n1) + r * (Is + h / 2 * m1) + d * (Is + h / 2 * m1)

        k3 = -(beta * (Is + h / 2 * m2) * (Ss + h / 2 * k2)) / N
        l3 = (beta * (Is + h / 2 * m2) * (Ss + h / 2 * k2)) / N - p * (Es + h / 2 * l2)
        m3 = p * (Es + h / 2 * l2) - q * (Is + h / 2 * m2)
        n3 = q * (Is + h / 2 * m2) - gamma * (Rs + h / 2 * n2)
        o3 = gamma * (Rs + h / 2 * n2) + r * (Is + h / 2 * m2) + d * (Is + h / 2 * m2)

        S.append(Ss + h * (k1 + k2 + k3) / 6)
        E.append(Es + h * (l1 + l2 + l3) / 6)
        I.append(Is + h * (m1 + m2 + m3) / 6)
        R.append(Rs + h * (n1 + n2 + n3) / 6)
        D.append(Ds + h * (o1 + o2 + o3) / 6)

    return S, E, I, R, D


# Example usage
beta = 0.3
gamma = 0.1
p = 0.1
q = 0.1
r = 0.1
d = 0.1
S0 = 1000
E0 = 10
I0 = 1
R0 = 0
D0 = 0
tf = 100
h = 0.1

S, E, I, R, D = serid_rk3(beta, gamma, p, q, r, d, S0, E0, I0, R0, D0, tf, h)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Dead')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SERID Model using RK3')
plt.show()

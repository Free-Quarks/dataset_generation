import matplotlib.pyplot as plt
import numpy as np

def sidarthe_euler(N, I0, D0, A0, R0, T0, H0, E0, days, beta, gamma, alpha, theta, delta, rho):
    S0 = N - I0 - D0 - A0 - R0 - T0 - H0 - E0

    S = [S0]
    I = [I0]
    D = [D0]
    A = [A0]
    R = [R0]
    T = [T0]
    H = [H0]
    E = [E0]

    dt = 0.1

    for day in range(days):
        s = S[-1]
        i = I[-1]
        d = D[-1]
        a = A[-1]
        r = R[-1]
        t = T[-1]
        h = H[-1]
        e = E[-1]

        dsdt = -beta * s * (i + alpha * a + theta * h) / N
        didt = (beta * s * (i + alpha * a + theta * h) / N) - (gamma + delta) * i
        dddt = delta * i
        dadt = (1 - rho) * gamma * i - alpha * a
        drdt = rho * gamma * i
        dtdt = rho * theta * h
        dhdt = rho * (1 - theta) * h
        dedt = delta * i

        S.append(s + dsdt * dt)
        I.append(i + didt * dt)
        D.append(d + dddt * dt)
        A.append(a + dadt * dt)
        R.append(r + drdt * dt)
        T.append(t + dtdt * dt)
        H.append(h + dhdt * dt)
        E.append(e + dedt * dt)

    x = np.linspace(0, days, int(days / dt) + 1)

    plt.plot(x, S, label='Susceptible')
    plt.plot(x, I, label='Infected')
    plt.plot(x, D, label='Deaths')
    plt.plot(x, A, label='Asymptomatic')
    plt.plot(x, R, label='Recovered')
    plt.plot(x, T, label='Tested')
    plt.plot(x, H, label='Hospitalized')
    plt.plot(x, E, label='Exposed')
    plt.xlabel('Days')
    plt.ylabel('Number of Individuals')
    plt.title('SIDARTHE Epidemiological Model using Euler Method')
    plt.legend()
    plt.show()

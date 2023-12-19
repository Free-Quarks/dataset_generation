```python
import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, t_max):
    def dS_dt(S, I, R, beta, gamma):
        return -beta * S * I / N

    def dI_dt(S, I, R, beta, gamma):
        return beta * S * I / N - gamma * I

    def dR_dt(S, I, R, beta, gamma):
        return gamma * I

    def RK4_step(S, I, R, beta, gamma, dt):
        dS1 = dS_dt(S, I, R, beta, gamma)
        dI1 = dI_dt(S, I, R, beta, gamma)
        dR1 = dR_dt(S, I, R, beta, gamma)

        dS2 = dS_dt(S + 0.5 * dt * dS1, I + 0.5 * dt * dI1, R + 0.5 * dt * dR1, beta, gamma)
        dI2 = dI_dt(S + 0.5 * dt * dS1, I + 0.5 * dt * dI1, R + 0.5 * dt * dR1, beta, gamma)
        dR2 = dR_dt(S + 0.5 * dt * dS1, I + 0.5 * dt * dI1, R + 0.5 * dt * dR1, beta, gamma)

        dS3 = dS_dt(S + 0.5 * dt * dS2, I + 0.5 * dt * dI2, R + 0.5 * dt * dR2, beta, gamma)
        dI3 = dI_dt(S + 0.5 * dt * dS2, I + 0.5 * dt * dI2, R + 0.5 * dt * dR2, beta, gamma)
        dR3 = dR_dt(S + 0.5 * dt * dS2, I + 0.5 * dt * dI2, R + 0.5 * dt * dR2, beta, gamma)

        dS4 = dS_dt(S + dt * dS3, I + dt * dI3, R + dt * dR3, beta, gamma)
        dI4 = dI_dt(S + dt * dS3, I + dt * dI3, R + dt * dR3, beta, gamma)
        dR4 = dR_dt(S + dt * dS3, I + dt * dI3, R + dt * dR3, beta, gamma)

        S += dt / 6 * (dS1 + 2 * dS2 + 2 * dS3 + dS4)
        I += dt / 6 * (dI1 + 2 * dI2 + 2 * dI3 + dI4)
        R += dt / 6 * (dR1 + 2 * dR2 + 2 * dR3 + dR4)

        return S, I, R

    S = N - I0 - R0
    I = I0
    R = R0
    t = 0
    dt = 0.1

    S_vals = [S]
    I_vals = [I]
    R_vals = [R]
    t_vals = [t]

    while t < t_max:
        S, I, R = RK4_step(S, I, R, beta, gamma, dt)
        t += dt

        S_vals.append(S)
        I_vals.append(I)
        R_vals.append(R)
        t_vals.append(t)

    plt.plot(t_vals, S_vals, label='Susceptible')
    plt.plot(t_vals, I_vals, label='Infected')
    plt.plot(t_vals, R_vals, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


SIR_model(0.3, 0.1, 1000, 1, 0, 100)
```

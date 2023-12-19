import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, N, I0, R0, t_end, dt):
    def derivs(SIR, t):
        S, I, R = SIR
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    S = N - I0 - R0
    I = I0
    R = R0
    t = np.arange(0, t_end+dt, dt)
    SIR = np.zeros((len(t), 3))
    SIR[0] = S, I, R

    for i in range(1, len(t)):
        k1 = derivs(SIR[i-1], t[i-1])
        k2 = derivs(SIR[i-1] + 0.5*dt*k1, t[i-1] + 0.5*dt)
        k3 = derivs(SIR[i-1] + 0.5*dt*k2, t[i-1] + 0.5*dt)
        k4 = derivs(SIR[i-1] + dt*k3, t[i-1] + dt)
        SIR[i] = SIR[i-1] + (1 / 6) * (k1 + 2*k2 + 2*k3 + k4) * dt

    return t, SIR[:, 0], SIR[:, 1], SIR[:, 2]

N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_end = 100
dt = 0.1


# Run the model

t, S, I, R = SIR_RK4(beta, gamma, N, I0, R0, t_end, dt)

# Plot the results

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()


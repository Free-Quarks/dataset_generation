import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def model(SIR, t, beta, gamma):
    S, I, R = SIR
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

S0 = 0.9
I0 = 0.1
R0 = 0.0
beta = 0.3
gamma = 0.1

t = np.linspace(0, 100, 10000)

SIR = odeint(model, [S0, I0, R0], t, args=(beta, gamma))

plt.figure(figsize=(6,4))
plt.plot(t, SIR[:,0], label='S(t)')
plt.plot(t, SIR[:,1], label='I(t)')
plt.plot(t, SIR[:,2], label='R(t)')
plt.legend()
plt.show()

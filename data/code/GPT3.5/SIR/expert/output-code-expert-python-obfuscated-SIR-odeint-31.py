```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0
beta, gamma = 0.2, 0.1
t = np.linspace(0, 100, 100)
y0 = [S0, I0, R0]

result = odeint(SIR_model, y0, t, args=(beta, gamma))

plt.plot(t, result[:, 0], label='S(t)')
plt.plot(t, result[:, 1], label='I(t)')
plt.plot(t, result[:, 2], label='R(t)')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
```

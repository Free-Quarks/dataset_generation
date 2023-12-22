import numpy as np
import matplotlib.pyplot as plt
import json

def RK4_SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def RK4(h, t, y, params):
    k1 = h * np.array(RK4_SIR_model(y, t, *params))
    k2 = h * np.array(RK4_SIR_model(y + k1/2, t + h/2, *params))
    k3 = h * np.array(RK4_SIR_model(y + k2/2, t + h/2, *params))
    k4 = h * np.array(RK4_SIR_model(y + k3, t + h, *params))
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

S0, I0, R0 = 999, 1, 0
beta, gamma = 0.2, 0.1
t = np.linspace(0, 160, 160)
y = [S0, I0, R0]
h = 1

for j in range(1, len(t)):
    y_next = RK4(h, t[j-1], y[-1], [beta, gamma])
    y.append(y_next)

S, I, R = np.hsplit(np.array(y), 3)
plt.figure(figsize=[6,4])
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.legend()
plt.xlabel('Time /days')
plt.ylabel('Number')
plt.grid()
plt.show()

output = {
    'code': open(__file__).read(),
    'function_name': 'RK4_SIR_model'
}
json_output = json.dumps(output)

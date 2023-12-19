import numpy as np
import matplotlib.pyplot as plt

def SIR_model(init_vals, params, t): 
    S_0, I_0, R_0 = init_vals
    S, I, R = [S_0], [I_0], [R_0]
    alpha, beta = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (alpha * S[-1] * I[-1]) * dt
        next_I = I[-1] + (alpha * S[-1] * I[-1] - beta * I[-1]) * dt
        next_R = R[-1] + (beta * I[-1]) * dt
        S.append(next_S)
        I.append(next_I)
        R.append(next_R)
    return S, I, R

# Example usage
alpha = 0.2
beta = 0.1
init_vals = 1 - alpha, alpha, 0
params = alpha, beta
t = np.linspace(0, 100, 100)
S, I, R = SIR_model(init_vals, params, t)

plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time')
plt.ylabel('Proportion of population')
plt.title('SIR Model')
plt.legend()
plt.show()

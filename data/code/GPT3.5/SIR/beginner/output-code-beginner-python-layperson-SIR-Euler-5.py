import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S0, I0, R0, T):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.linspace(0, T, T+1)
    dt = t[1] - t[0]
    for i in range(T):
        dS = -beta * S[i] * I[i] / N
        dI = beta * S[i] * I[i] / N - gamma * I[i]
        dR = gamma * I[i]
        S.append(S[i] + dS * dt)
        I.append(I[i] + dI * dt)
        R.append(R[i] + dR * dt)
    return t, S, I, R


# Parameters
beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
T = 100

# Run the model
t, S, I, R = SIR_model(beta, gamma, S0, I0, R0, T)

# Plotting
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()


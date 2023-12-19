import numpy as np
import matplotlib.pyplot as plt

# Function to implement SIR model

def SIR_model(S, I, R, beta, gamma, N, days):
    S_values = [S]
    I_values = [I]
    R_values = [R]
    t_values = [0]

    # Euler's method
    dt = 0.1
    for day in range(days):
        dS = -beta * S * I / N
        dI = (beta * S * I / N) - gamma * I
        dR = gamma * I

        S += dS * dt
        I += dI * dt
        R += dR * dt
        t = t_values[-1] + dt

        S_values.append(S)
        I_values.append(I)
        R_values.append(R)
        t_values.append(t)

    return (S_values, I_values, R_values, t_values)


# Initial values
S = 9999
I = 1
R = 0
beta = 0.3
gamma = 0.1
N = S + I + R

days = 100

# Run simulation
S_values, I_values, R_values, t_values = SIR_model(S, I, R, beta, gamma, N, days)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t_values, S_values, label='Susceptible')
plt.plot(t_values, I_values, label='Infected')
plt.plot(t_values, R_values, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(S, I, R, beta, gamma):
    S_prime = -beta * S * I
    I_prime = beta * S * I - gamma * I
    R_prime = gamma * I
    return S_prime, I_prime, R_prime

def rk2(S, I, R, beta, gamma, dt):
    S_prime, I_prime, R_prime = sir_model(S, I, R, beta, gamma)
    S_half = S + 0.5 * dt * S_prime
    I_half = I + 0.5 * dt * I_prime
    R_half = R + 0.5 * dt * R_prime
    S_prime_half, I_prime_half, R_prime_half = sir_model(S_half, I_half, R_half, beta, gamma)
    S_next = S + dt * S_prime_half
    I_next = I + dt * I_prime_half
    R_next = R + dt * R_prime_half
    return S_next, I_next, R_next

def sir_simulation(S0, I0, R0, beta, gamma, dt, T):
    S, I, R = S0, I0, R0
    S_list, I_list, R_list = [S0], [I0], [R0]
    for _ in np.arange(0, T, dt):
        S, I, R = rk2(S, I, R, beta, gamma, dt)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
    return S_list, I_list, R_list

def plot_sir_simulation(S, I, R, dt):
    t = np.arange(0, len(S)*dt, dt)
    plt.figure(figsize=(10,6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()

# parameters for the model
S0 = 0.99
I0 = 0.01
R0 = 0.0
beta = 1.0
gamma = 0.5
dt = 0.01
T = 10.0

# simulate the SIR model
S, I, R = sir_simulation(S0, I0, R0, beta, gamma, dt, T)

# plot the result
plot_sir_simulation(S, I, R, dt)
plt.show()

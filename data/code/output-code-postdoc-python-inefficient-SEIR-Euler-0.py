import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, I0, R0, T):
    # Initial conditions
    S0 = N - I0 - R0
    E0 = 0

    # Arrays to store results
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]

    # Euler method
    dt = 1
    t = np.arange(0, T, dt)

    for _ in t:
        S.append(S[-1] - (beta * S[-1] * I[-1] / N) * dt)
        E.append(E[-1] + (beta * S[-1] * I[-1] / N - sigma * E[-1]) * dt)
        I.append(I[-1] + (sigma * E[-1] - gamma * I[-1]) * dt)
        R.append(R[-1] + (gamma * I[-1]) * dt)

    return S, E, I, R

# Example usage
def plot_seir(S, E, I, R):
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='Susceptible')
    plt.plot(E, label='Exposed')
    plt.plot(I, label='Infected')
    plt.plot(R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()


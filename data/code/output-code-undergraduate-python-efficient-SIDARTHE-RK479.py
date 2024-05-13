import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(t, y, beta, sigma, alpha, rho, theta, eta):
    S, I, D, A, R, T, H, E = y
    N = sum(y)
    dS = -beta*S*I/N
    dI = beta*S*I/N - sigma*I - alpha*I - rho*T
    dD = theta*rho*T - eta*D
    dA = alpha*I - sigma*A
    dR = sigma*(I + A)
    dT = (1-theta)*rho*T
    dH = eta*D
    dE = sigma*A
    return [dS, dI, dD, dA, dR, dT, dH, dE]


def run_sidarthe_model(beta, sigma, alpha, rho, theta, eta, initial_conditions, t_max):
    t = np.linspace(0, t_max, t_max+1)
    y = np.zeros((t_max+1, len(initial_conditions)))
    y[0] = initial_conditions

    for i in range(t_max):
        k1 = sidarthe_model(t[i], y[i], beta, sigma, alpha, rho, theta, eta)
        k2 = sidarthe_model(t[i]+0.5, y[i]+0.5*k1, beta, sigma, alpha, rho, theta, eta)
        k3 = sidarthe_model(t[i]+0.5, y[i]+0.5*k2, beta, sigma, alpha, rho, theta, eta)
        k4 = sidarthe_model(t[i]+1, y[i]+k3, beta, sigma, alpha, rho, theta, eta)
        y[i+1] = y[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t, y


beta = 0.3
sigma = 0.05
alpha = 0.02
rho = 0.2
theta = 0.5
eta = 0.1
initial_conditions = [10**6, 100, 10, 10, 0, 0, 0, 0]
t_max = 100

t, y = run_sidarthe_model(beta, sigma, alpha, rho, theta, eta, initial_conditions, t_max)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(t, y[:, 0], label='Susceptible')
plt.plot(t, y[:, 1], label='Infected')
plt.plot(t, y[:, 2], label='Deceased')
plt.plot(t, y[:, 3], label='Asymptomatic')
plt.plot(t, y[:, 4], label='Recovered')
plt.plot(t, y[:, 5], label='Tested')
plt.plot(t, y[:, 6], label='Hospitalized')
plt.plot(t, y[:, 7], label='Exposed')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.legend()
plt.grid(True)
plt.show()

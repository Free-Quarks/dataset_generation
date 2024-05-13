import numpy as np
import matplotlib.pyplot as plt


def sidarthe(init, alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, mu, nu, xi, rho, sigma, tau, upsilon, phi, chi, psi, omega, T):
    # Initialize arrays
    S = np.zeros(T+1)
    I = np.zeros(T+1)
    D = np.zeros(T+1)
    A = np.zeros(T+1)
    R = np.zeros(T+1)
    T = np.zeros(T+1)
    H = np.zeros(T+1)
    E = np.zeros(T+1)

    # Initial conditions
    S[0], I[0], D[0], A[0], R[0], T[0], H[0], E[0] = init

    # Time step
    dt = 0.1

    # Runge-Kutta integration
    for t in range(T):
        # Calculate derivatives
        dSdt = -alpha*S[t]*(I[t]+theta*A[t])
        dIdt = alpha*S[t]*(I[t]+theta*A[t])-beta*I[t]
        dDdt = xi*iota*I[t]
        dAdt = (1-xi)*iota*I[t]-gamma*A[t]
        dRdt = gamma*A[t]
        dTdt = epsilon*delta*I[t]-mu*T[t]
        dHdt = mu*T[t]-nu*eta*H[t]
        dEdt = nu*eta*H[t]-rho*epsilon*E[t]-phi*chi*E[t]+psi*T[t]

        # Update variables using RK3
        S[t+1] = S[t] + dt*(dSdt+2*dSdt1+2*dSdt2+dSdt3)/6
        I[t+1] = I[t] + dt*(dIdt+2*dIdt1+2*dIdt2+dIdt3)/6
        D[t+1] = D[t] + dt*(dDdt+2*dDdt1+2*dDdt2+dDdt3)/6
        A[t+1] = A[t] + dt*(dAdt+2*dAdt1+2*dAdt2+dAdt3)/6
        R[t+1] = R[t] + dt*(dRdt+2*dRdt1+2*dRdt2+dRdt3)/6
        T[t+1] = T[t] + dt*(dTdt+2*dTdt1+2*dTdt2+dTdt3)/6
        H[t+1] = H[t] + dt*(dHdt+2*dHdt1+2*dHdt2+dHdt3)/6
        E[t+1] = E[t] + dt*(dEdt+2*dEdt1+2*dEdt2+dEdt3)/6

    return S, I, D, A, R, T, H, E


# Set initial conditions
init = [9999, 1, 0, 0, 0, 0, 0, 0]

# Set model parameters
alpha = 0.3
beta = 0.05
gamma = 0.1
theta = 0.05
iota = 0.3
xi = 0.2
epsilon = 0.1
delta = 0.05
mu = 0.1
nu = 0.1
rho = 0.1
sigma = 0.1
tau = 0.1
upsilon = 0.1
phi = 0.1
chi = 0.1
psi = 0.1
omega = 0.1
T = 100

# Run SIDARTHE model
S, I, D, A, R, T, H, E = sidarthe(init, alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, mu, nu, xi, rho, sigma, tau, upsilon, phi, chi, psi, omega, T)

# Plot results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(D, label='Deceased')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Tested')
plt.plot(H, label='Hospitalized')
plt.plot(E, label='Exposure')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.show()

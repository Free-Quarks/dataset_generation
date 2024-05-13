import numpy as np
import matplotlib.pyplot as plt

# Function implementing the SIDARTHE model

def sidarthe_model(beta, gamma, delta, alpha, rho, theta, epsilon, sigma, t, s0, i0, d0, a0, r0, t0):
    N = s0 + i0 + d0 + a0 + r0 + t0
    S = [s0]
    I = [i0]
    D = [d0]
    A = [a0]
    R = [r0]
    T = [t0]

    dt = t[1] - t[0]

    for j in range(len(t)-1):
        S.append(S[j] - beta*S[j]*I[j]/N - delta*S[j]*A[j]/N - alpha*S[j]*T[j]/N)
        I.append(I[j] + beta*S[j]*I[j]/N + epsilon*R[j] - gamma*I[j] - rho*I[j] - theta*I[j])
        D.append(D[j] + delta*S[j]*A[j]/N)
        A.append(A[j] + alpha*S[j]*T[j]/N - sigma*A[j])
        R.append(R[j] + gamma*I[j] - epsilon*R[j])
        T.append(T[j] + rho*I[j] - theta*I[j])

    return S, I, D, A, R, T


# Parameters
beta = 0.5
gamma = 0.1
alpha = 0.1
rho = 0.01
delta = 0.03
theta = 0.01
epsilon = 0.02
sigma = 0.02
t = np.linspace(0, 100, 1001)
s0 = 500
i0 = 50
d0 = 10
a0 = 10
r0 = 0
t0 = 0

# Run the model
S, I, D, A, R, T = sidarthe_model(beta, gamma, delta, alpha, rho, theta, epsilon, sigma, t, s0, i0, d0, a0, r0, t0)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Deceased')
plt.plot(t, A, label='Affected')
plt.plot(t, R, label='Recovered')
plt.plot(t, T, label='Tested')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()

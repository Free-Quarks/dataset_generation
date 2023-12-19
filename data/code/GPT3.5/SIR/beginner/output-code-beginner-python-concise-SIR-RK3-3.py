import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, tmax):
    def SIRmodel(SIR, t, beta, gamma):
        S, I, R = SIR
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    # Initialize arrays
    t = np.linspace(0, tmax, tmax+1)
    S = np.zeros(tmax+1)
    I = np.zeros(tmax+1)
    R = np.zeros(tmax+1)
    
    # Set initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    # Solve the system of differential equations using the Runge-Kutta 3rd order method
    for i in range(tmax):
        h = t[i+1] - t[i]
        k1 = SIRmodel([S[i], I[i], R[i]], t[i], beta, gamma)
        k2 = SIRmodel([S[i] + h/2 * k1[0], I[i] + h/2 * k1[1], R[i] + h/2 * k1[2]], t[i] + h/2, beta, gamma)
        k3 = SIRmodel([S[i] - h * k1[0] + 2 * h * k2[0], I[i] - h * k1[1] + 2 * h * k2[1], R[i] - h * k1[2] + 2 * h * k2[2]], t[i] + h, beta, gamma)
        S[i+1] = S[i] + h/6 * (k1[0] + 4 * k2[0] + k3[0])
        I[i+1] = I[i] + h/6 * (k1[1] + 4 * k2[1] + k3[1])
        R[i+1] = R[i] + h/6 * (k1[2] + 4 * k2[2] + k3[2])
    
    # Return the solution
    return S, I, R

# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
tmax = 100
S, I, R = SIR_RK3(beta, gamma, N, I0, tmax)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model with RK3')
plt.legend()
plt.show()

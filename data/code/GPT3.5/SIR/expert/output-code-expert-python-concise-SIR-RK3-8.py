import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(N, beta, gamma, I0, T):
    # Initial conditions
    S0 = N - I0
    R0 = 0
    Y0 = np.array([S0, I0, R0])
    
    # Time vector
    t = np.linspace(0, T, T+1)
    dt = t[1] - t[0]
    
    # Function defining the system of equations
    def system(Y, t):
        S, I, R = Y
        dS = -beta*S*I/N
        dI = beta*S*I/N - gamma*I
        dR = gamma*I
        return np.array([dS, dI, dR])
    
    # Runge-Kutta 3rd order method to solve the system of equations
    def runge_kutta(Y, t, dt):
        k1 = system(Y, t)
        k2 = system(Y+0.5*dt*k1, t+0.5*dt)
        k3 = system(Y-dt*k1+2*dt*k2, t+dt)
        return Y + (dt/6)*(k1 + 4*k2 + k3)
    
    # Solve the system of equations
    Y = Y0
    S = [Y0[0]]
    I = [Y0[1]]
    R = [Y0[2]]
    for i in range(T):
        Y = runge_kutta(Y, t[i], dt)
        S.append(Y[0])
        I.append(Y[1])
        R.append(Y[2])
    
    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model: RK3')
    plt.legend()
    plt.show()
    
    return


# Example usage
N = 1000
beta = 0.2
gamma = 0.1
I0 = 1
T = 100

SIR_RK3(N, beta, gamma, I0, T)

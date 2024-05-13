import numpy as np
import matplotlib.pyplot as plt


def serid_euler(beta, gamma_1, gamma_2, N, I0, R0, D0, T):
    # Set initial conditions
    S0 = N - I0 - R0 - D0
    I = [I0]
    R = [R0]
    D = [D0]
    S = [S0]
    t = np.linspace(0, T, T+1)
    dt = t[1] - t[0]
    
    # Euler's method
    for i in range(T):
        dS = -beta * I[i] * S[i] / N
        dI = (beta * I[i] * S[i] / N) - (gamma_1 * I[i]) - (gamma_2 * I[i])
        dR = gamma_1 * I[i]
        dD = gamma_2 * I[i]
        
        S.append(S[i] + dS * dt)
        I.append(I[i] + dI * dt)
        R.append(R[i] + dR * dt)
        D.append(D[i] + dD * dt)
    
    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Deceased')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SERID Model Simulation')
    plt.legend()
    plt.show()
}


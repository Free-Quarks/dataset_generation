import numpy as np
import matplotlib.pyplot as plt


def serid_model(N, d, beta, gamma, delta, epsilon, mu, rho, sigma, t_max):
    # Initial conditions
    S_0 = N - 1
    E_0 = 1
    I_0 = 0
    R_0 = 0
    D_0 = 0
    
    # Time vector
    t = np.linspace(0, t_max, t_max+1)
    
    # Compartment arrays
    S = np.zeros(t_max+1)
    E = np.zeros(t_max+1)
    I = np.zeros(t_max+1)
    R = np.zeros(t_max+1)
    D = np.zeros(t_max+1)
    
    S[0] = S_0
    E[0] = E_0
    I[0] = I_0
    R[0] = R_0
    D[0] = D_0
    
    # RK4 solver
    for i in range(t_max):
        h = t[i+1] - t[i]
        k1 = d * (beta * S[i] * I[i] + epsilon * S[i] * E[i])
        l1 = gamma * I[i]
        m1 = delta * E[i]
        n1 = rho * I[i]
        o1 = sigma * I[i]
        p1 = mu * I[i]
        
        k2 = d * (beta * (S[i] + 0.5 * h * k1) * (I[i] + 0.5 * h * l1) + epsilon * (S[i] + 0.5 * h * k1) * (E[i] + 0.5 * h * m1))
        l2 = gamma * (I[i] + 0.5 * h * l1)
        m2 = delta * (E[i] + 0.5 * h * m1)
        n2 = rho * (I[i] + 0.5 * h * l1)
        o2 = sigma * (I[i] + 0.5 * h * l1)
        p2 = mu * (I[i] + 0.5 * h * l1)
        
        k3 = d * (beta * (S[i] + 0.5 * h * k2) * (I[i] + 0.5 * h * l2) + epsilon * (S[i] + 0.5 * h * k2) * (E[i] + 0.5 * h * m2))
        l3 = gamma * (I[i] + 0.5 * h * l2)
        m3 = delta * (E[i] + 0.5 * h * m2)
        n3 = rho * (I[i] + 0.5 * h * l2)
        o3 = sigma * (I[i] + 0.5 * h * l2)
        p3 = mu * (I[i] + 0.5 * h * l2)
        
        k4 = d * (beta * (S[i] + h * k3) * (I[i] + h * l3) + epsilon * (S[i] + h * k3) * (E[i] + h * m3))
        l4 = gamma * (I[i] + h * l3)
        m4 = delta * (E[i] + h * m3)
        n4 = rho * (I[i] + h * l3)
        o4 = sigma * (I[i] + h * l3)
        p4 = mu * (I[i] + h * l3)
        
        S[i+1] = S[i] - (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        E[i+1] = E[i] + h * (k1 + 2*k2 + 2*k3 + k4)
        I[i+1] = I[i] + h * (l1 + 2*l2 + 2*l3 + l4)
        R[i+1] = R[i] + h * (n1 + 2*n2 + 2*n3 + n4)
        D[i+1] = D[i] + h * (o1 + 2*o2 + 2*o3 + o4 + p1 + 2*p2 + 2*p3 + p4)
        
    # Plotting
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Deceased')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SERID Model')
    plt.legend()
    plt.show()
}


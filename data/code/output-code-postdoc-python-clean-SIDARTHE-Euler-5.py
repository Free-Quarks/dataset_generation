import numpy as np


def sidarthe_model(population, initial_conditions, parameters, time):
    S, I, D, A, R, T, H, E = initial_conditions
    beta, alpha, delta, gamma, rho, theta, epsilon = parameters
    N = population
    dt = time[1] - time[0]
    num_steps = len(time)
    num_compartments = len(initial_conditions)
    
    S_values = np.zeros(num_steps)
    I_values = np.zeros(num_steps)
    D_values = np.zeros(num_steps)
    A_values = np.zeros(num_steps)
    R_values = np.zeros(num_steps)
    T_values = np.zeros(num_steps)
    H_values = np.zeros(num_steps)
    E_values = np.zeros(num_steps)
    
    S_values[0] = S
    I_values[0] = I
    D_values[0] = D
    A_values[0] = A
    R_values[0] = R
    T_values[0] = T
    H_values[0] = H
    E_values[0] = E
    
    for i in range(1, num_steps):
        dS = -beta * S * (I + alpha * A) / N
        dI = beta * S * (I + alpha * A) / N - (1 - delta) * gamma * I - delta * rho * I
        dD = delta * gamma * I
        dA = delta * rho * I - theta * E - epsilon * A
        dR = (1 - delta) * gamma * I
        dT = theta * E
        dH = epsilon * A
        dE = 0
        
        S += dt * dS
        I += dt * dI
        D += dt * dD
        A += dt * dA
        R += dt * dR
        T += dt * dT
        H += dt * dH
        E += dt * dE
        
        S_values[i] = S
        I_values[i] = I
        D_values[i] = D
        A_values[i] = A
        R_values[i] = R
        T_values[i] = T
        H_values[i] = H
        E_values[i] = E
    
    return S_values, I_values, D_values, A_values, R_values, T_values, H_values, E_values
}

import numpy as np

def serid_rk4(beta, gamma, mu, N, I0, S0, R0, D0, t_max, h):
    # Define the system of differential equations
    def system(t, y):
        S, E, R, I, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - gamma * E - mu * E
        dRdt = gamma * E
        dIdt = mu * E - gamma * I - D
        dDdt = gamma * I
        return [dSdt, dEdt, dRdt, dIdt, dDdt]
    
    # Initialize the lists for storing the results
    t_list = np.arange(0, t_max + h, h)
    S_list = np.zeros(len(t_list))
    E_list = np.zeros(len(t_list))
    R_list = np.zeros(len(t_list))
    I_list = np.zeros(len(t_list))
    D_list = np.zeros(len(t_list))
    
    # Set the initial conditions
    S_list[0] = S0
    E_list[0] = 0
    R_list[0] = R0
    I_list[0] = I0
    D_list[0] = D0
    
    # Perform the RK4 integration
    for i in range(1, len(t_list)):
        t = t_list[i-1]
        y = [S_list[i-1], E_list[i-1], R_list[i-1], I_list[i-1], D_list[i-1]]
        k1 = h * system(t, y)
        k2 = h * system(t + h/2, y + k1/2)
        k3 = h * system(t + h/2, y + k2/2)
        k4 = h * system(t + h, y + k3)
        y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        S_list[i] = y_next[0]
        E_list[i] = y_next[1]
        R_list[i] = y_next[2]
        I_list[i] = y_next[3]
        D_list[i] = y_next[4]
    
    # Return the results
    return t_list, S_list, E_list, R_list, I_list, D_list


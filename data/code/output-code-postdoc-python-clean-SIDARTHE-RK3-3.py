import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(beta, gamma, delta, alpha, rho, theta, population, initial_conditions, time):
    S0, I0, D0, A0, R0, T0, H0, E0 = initial_conditions
    N = np.sum(initial_conditions)
    
    S = [S0]
    I = [I0]
    D = [D0]
    A = [A0]
    R = [R0]
    T = [T0]
    H = [H0]
    E = [E0]
    
    dt = time[1] - time[0]
    
    for t in range(1, len(time)):
        dSdt = -beta * S[-1] * I[-1] / N
        dIdt = beta * S[-1] * I[-1] / N - (gamma + delta + alpha) * I[-1]
        dDdt = delta * I[-1] - (theta + rho) * D[-1]
        dAdt = alpha * I[-1] - (rho + gamma) * A[-1]
        dRdt = gamma * (I[-1] + A[-1]) + rho * (D[-1])
        dTdt = theta * D[-1]
        dHdt = rho * A[-1]
        dEdt = gamma * I[-1]
        
        S.append(S[-1] + dt * dSdt)
        I.append(I[-1] + dt * dIdt)
        D.append(D[-1] + dt * dDdt)
        A.append(A[-1] + dt * dAdt)
        R.append(R[-1] + dt * dRdt)
        T.append(T[-1] + dt * dTdt)
        H.append(H[-1] + dt * dHdt)
        E.append(E[-1] + dt * dEdt)
    
    return S, I, D, A, R, T, H, E
}


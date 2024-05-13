import numpy as np

def sidarthe_rk3(initial_conditions, parameters, t_start, t_end, t_step):
    def sidarthe(t, y, params):
        S, I, D, A, R, T, H, E = y
        beta, sigma, gamma, alpha, theta, delta, epsilon = params
        N = S + I + D + A + R + T + H + E
        dSdt = -beta * S * (I + alpha * A) / N
        dIdt = (beta * S * (I + alpha * A) / N) - (sigma * I)
        dDdt = delta * sigma * I - gamma * D
        dAdt = (1 - delta) * sigma * I - theta * A
        dRdt = gamma * D
        dTdt = theta * A
        dHdt = epsilon * sigma * I
        dEdt = sigma * I - alpha * epsilon * sigma * I
        return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]
    
    y = initial_conditions
    t = np.arange(t_start, t_end + t_step, t_step)
    
    S, I, D, A, R, T, H, E = [], [], [], [], [], [], [], []
    
    for i, _ in enumerate(t):
        S.append(y[0])
        I.append(y[1])
        D.append(y[2])
        A.append(y[3])
        R.append(y[4])
        T.append(y[5])
        H.append(y[6])
        E.append(y[7])
        k1 = t_step * np.array(sidarthe(t[i], y, parameters))
        k2 = t_step * np.array(sidarthe(t[i] + t_step / 3, y + k1 / 3, parameters))
        k3 = t_step * np.array(sidarthe(t[i] + 2 * t_step / 3, y - k1 / 3 + k2, parameters))
        y = y + (k1 + 3 * k3) / 4
    
    return {'S': S, 'I': I, 'D': D, 'A': A, 'R': R, 'T': T, 'H': H, 'E': E}

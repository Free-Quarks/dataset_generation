def serid_model(t, y, beta, gamma, alpha, N):
    S, E, R, I, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - alpha * E
    dRdt = gamma * I
    dIdt = alpha * E - gamma * I - D
    dDdt = D
    return dSdt, dEdt, dRdt, dIdt, dDdt


def run_serid_model(beta, gamma, alpha, N, S0, E0, R0, I0, D0, days):
    import numpy as np
    from scipy.integrate import odeint
    
    y0 = S0, E0, R0, I0, D0
    t = np.linspace(0, days, days)
    
    result = odeint(serid_model, y0, t, args=(beta, gamma, alpha, N))
    
    S, E, R, I, D = result.T
    
    return S, E, R, I, D

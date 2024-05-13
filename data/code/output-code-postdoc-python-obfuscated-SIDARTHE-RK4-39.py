import numpy as np


def sidarthe_model(y, t, p):
    S, I, D, A, R, T, H, E = y
    beta, gamma, delta, alpha, theta, mu, eps = p
    N = S + I + D + A + R + T + H + E
    dSdt = -beta*S*(I + alpha*A + delta*T + theta*H + mu*E)/N
    dIdt = beta*S*(I + alpha*A + delta*T + theta*H + mu*E)/N - gamma*I
    dDdt = gamma*I
    dAdt = (1-eps)*gamma*I - alpha*A
    dRdt = eps*gamma*I
    dTdt = delta*T - theta*T
    dHdt = theta*T - mu*E
    dEdt = alpha*A + mu*E
    return np.array([dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt])



def sidarthe_simulation(S, I, D, A, R, T, H, E, beta, gamma, delta, alpha, theta, mu, eps, t_end, n_points):
    y0 = np.array([S, I, D, A, R, T, H, E])
    p = np.array([beta, gamma, delta, alpha, theta, mu, eps])
    t = np.linspace(0, t_end, n_points)
    y = odeint(sidarthe_model, y0, t, args=(p,))
    return t, y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6], y[:,7]

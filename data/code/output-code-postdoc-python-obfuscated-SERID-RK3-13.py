from numpy import array
from scipy.integrate import solve_ivp


def serid_model(t, y, params):
    S, E, I, R, D = y
    N = sum(y)
    alpha, beta, gamma, delta, omega = params
    dSdt = - alpha * S * I / N
    dEdt = alpha * S * I / N - beta * E
    dIdt = beta * E - gamma * I - delta * I
    dRdt = gamma * I
    dDdt = delta * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def serid_simulation(initial_conditions, params, t_start, t_end, t_step):
    t_span = (t_start, t_end)
    t_eval = list(range(t_start, t_end + 1, t_step))
    sol = solve_ivp(serid_model, t_span, initial_conditions, args=(params,), method='RK45', t_eval=t_eval)
    return sol

import numpy as np
from scipy.integrate import odeint
def serid_model(y, t, p):
	S, E, I, R, D = y
	beta, sigma, gamma, mu = p
	dSdt = -beta * S * (I + E) + mu * R
	dEdt = beta * S * (I + E) - sigma * E
	dIdt = sigma * E - gamma * I - mu * I
	dRdt = gamma * I - mu * R
	dDdt = mu * (I + R)
	return [dSdt, dEdt, dIdt, dRdt, dDdt]

def simulate_serid_model(y0, t, p):
	solution = odeint(serid_model, y0, t, args=(p,))
	return solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3], solution[:, 4]

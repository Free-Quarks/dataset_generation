import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def serid_model(y, t, beta, gamma, delta):
	S, E, I, R, D = y
	dSdt = -beta * S * I
	dEdt = beta * S * I - delta * E
	dIdt = delta * E - gamma * I
	dRdt = gamma * I
	dDdt = delta * E
	return [dSdt, dEdt, dIdt, dRdt, dDdt]


def simulate_serid(S0, E0, I0, R0, D0, beta, gamma, delta, t_max):
	# Initial conditions
	init_cond = [S0, E0, I0, R0, D0]

	# Time points
	t = np.linspace(0, t_max, t_max + 1)

	# Simulate the ODE system
	solution = odeint(serid_model, init_cond, t, args=(beta, gamma, delta))

	# Plot the results
	plt.plot(t, solution[:, 0], label='S')
	plt.plot(t, solution[:, 1], label='E')
	plt.plot(t, solution[:, 2], label='I')
	plt.plot(t, solution[:, 3], label='R')
	plt.plot(t, solution[:, 4], label='D')

	plt.xlabel('Time')
	plt.ylabel('Population')
	plt.title('SERID Model')
	plt.legend()
	plt.show()


# Example usage
simulate_serid(1000, 10, 1, 0, 0, 0.2, 0.1, 0.05, 100)

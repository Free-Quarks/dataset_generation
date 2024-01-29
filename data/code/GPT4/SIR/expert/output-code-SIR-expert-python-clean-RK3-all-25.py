'import numpy as np\nimport matplotlib.pyplot as plt\nfrom scipy.integrate import solve_ivp\n\ndef sir_model(t, y, N, beta, gamma):\n    S, I, R = y\n    dSdt = -beta * S * I / N\n    dIdt = beta * S * I / N - gamma * I\n    dRdt = gamma * I\n    return [dSdt, dIdt, dRdt]\n\nN = 1000\nI0 = 1\nR0 = 0\nS0 = N - I0 - R0\nbeta = 0.2\ngamma = 0.1\n\nsol = solve_ivp(sir_model, [0, 160], [S0, I0, R0], args=(N, beta, gamma), dense_output=True)\n\nt = np.linspace(0, 160, 160)\nS, I, R = sol.sol(t)\nplt.figure(figsize=[6,4])\nplt.plot(t, S, 'b', label='Susceptible')\nplt.plot(t, I, 'r', label='Infected')\nplt.plot(t, R, 'g', label='Recovered')\nplt.legend()\nplt.xlabel('Time /days')\nplt.ylabel('Number (1000s)')\nplt.title('SIR model with RK3')\nplt.grid(True)\nplt.show()'
import numpy as np,matplotlib.pyplot as plt,scipy.integrate as spi
def dSEIR_dT(Y,t,beta,gamma):
 S,E,I,R = Y
 N = S+E+I+R
 dS = -beta*S*I/N
 dI = beta*S*I/N-gamma*I
 dR = gamma*I
 return dS,dI,dR
S0,E0,I0,R0 = 990,0,10,0
beta,gamma = 0.7,0.2
t = np.linspace(0,100,1000)
solution = spi.odeint(dSEIR_dT, (S0,E0,I0,R0), t, args=(beta, gamma))
plt.plot(t, solution[:, 0], label="S(t)")
plt.plot(t, solution[:, 1], label="I(t)")
plt.plot(t, solution[:, 2], label="R(t)")
plt.grid()
plt.legend()
plt.xlabel("Time")
plt.ylabel("Proportions")
plt.title("SIR model")
plt.show()

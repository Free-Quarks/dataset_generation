import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(R0, T_inf, T_hosp, T_icu, T_rec, T_die, N, E0, I0, R0, D0, A0, T0, H0, IC0):
    def diff_eqs(y, t):
        S, I, D, A, R, T, H, IC = y
        dS = -R0/T_inf * I * S / N
        dE = R0/T_inf * I * S / N -  A / (T_inf + T_hosp)
        dI = A / (T_inf + T_hosp) - I / T_rec
        dD = I / T_die
        dA = A / T_inf - A / (T_inf + T_hosp)
        dR = I / T_rec - R / T_rec
        dT = A / T_inf + I / T_rec - T / T_rec
        dH = I / T_rec - H / T_rec
        dIC = H / T_rec - IC / T_rec
        return dS, dI, dD, dA, dR, dT, dH, dIC
    y0 = S0, I0, D0, A0, R0, T0, H0, IC0
    t = np.linspace(0, 365, 365)
    result = odeint(diff_eqs, y0, t)
    S, I, D, A, R, T, H, IC = result.T
    return S, I, D, A, R, T, H, IC


R0 = 2.5
T_inf = 5
T_hosp = 10
T_icu = 14
T_rec = 21
T_die = 20
N = 1000000
E0 = 0
I0 = 1
R0 = 0
D0 = 0
A0 = 0
T0 = 0
H0 = 0
IC0 = 0

S, I, D, A, R, T, H, IC = sidarthe_model(R0, T_inf, T_hosp, T_icu, T_rec, T_die, N, E0, I0, R0, D0, A0, T0, H0, IC0)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Deceased')
plt.plot(t, A, label='Asymptomatic')
plt.plot(t, R, label='Recovered')
plt.plot(t, T, label='Total Active')
plt.plot(t, H, label='Hospitalized')
plt.plot(t, IC, label='ICU')
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()

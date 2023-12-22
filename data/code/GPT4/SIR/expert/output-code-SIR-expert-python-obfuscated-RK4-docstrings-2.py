import json
import matplotlib.pyplot as plt
import numpy as np

def _a(S, I, R, b, g):
    return -b*S*I

def _b(S, I, R, b, g):
    return b*S*I - g*I

def _c(S, I, R, b, g):
    return g*I

def _rk4(S, I, R, b, g, h):
    S1, I1, R1 = h*_a(S, I, R, b, g), h*_b(S, I, R, b, g), h*_c(S, I, R, b, g)
    S2, I2, R2 = h*_a(S+S1/2, I+I1/2, R+R1/2, b, g), h*_b(S+S1/2, I+I1/2, R+R1/2, b, g), h*_c(S+S1/2, I+I1/2, R+R1/2, b, g)
    S3, I3, R3 = h*_a(S+S2/2, I+I2/2, R+R2/2, b, g), h*_b(S+S2/2, I+I2/2, R+R2/2, b, g), h*_c(S+S2/2, I+I2/2, R+R2/2, b, g)
    S4, I4, R4 = h*_a(S+S3, I+I3, R+R3, b, g), h*_b(S+S3, I+I3, R+R3, b, g), h*_c(S+S3, I+I3, R+R3, b, g)
    return S + (S1 + 2*S2 + 2*S3 + S4)/6, I + (I1 + 2*I2 + 2*I3 + I4)/6, R + (R1 + 2*R2 + 2*R3 + R4)/6

def _sir(S0, I0, R0, b, g, T, h=0.1):
    S, I, R = S0, I0, R0
    Ss, Is, Rs = [S0], [I0], [R0]
    for _ in np.arange(0, T, h):
        S, I, R = _rk4(S, I, R, b, g, h)
        Ss.append(S)
        Is.append(I)
        Rs.append(R)
    return Ss, Is, Rs

def _plot(S, I, R):
    plt.plot(S, label="Susceptible")
    plt.plot(I, label="Infected")
    plt.plot(R, label="Recovered")
    plt.legend()
    plt.show()

S0, I0, R0 = 0.99, 0.01, 0
b, g = 0.5, 0.1
T = 100
S, I, R = _sir(S0, I0, R0, b, g, T)
_plot(S, I, R)

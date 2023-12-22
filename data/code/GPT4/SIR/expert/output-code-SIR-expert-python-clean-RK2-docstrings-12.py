import numpy as np
import matplotlib.pyplot as plt
import json

def RK2_SIR(y, t, dt, beta, gamma):
    S, I, R = y
    f1 = lambda S,I,R : -beta*S*I
    f2 = lambda S,I,R : beta*S*I - gamma*I
    f3 = lambda S,I,R : gamma*I

    k1_S = dt * f1(S, I, R)
    k1_I = dt * f2(S, I, R)
    k1_R = dt * f3(S, I, R)

    k2_S = dt * f1(S + 0.5*k1_S, I + 0.5*k1_I, R + 0.5*k1_R)
    k2_I = dt * f2(S + 0.5*k1_S, I + 0.5*k1_I, R + 0.5*k1_R)
    k2_R = dt * f3(S + 0.5*k1_S, I + 0.5*k1_I, R + 0.5*k1_R)

    S += k2_S
    I += k2_I
    R += k2_R
    
    y = [S, I, R]

    return y

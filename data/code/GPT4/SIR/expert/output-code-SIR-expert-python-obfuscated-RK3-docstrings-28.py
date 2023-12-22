import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
def _xyz(S, I, R, b, g, t):
    _b,_c,_d=b*S*I,b*I-g*S,b*S+g*I
    _e=_b+_d
    _f=_e-_c
    _g=2*_b-_f
    _h=(_b+_g)/3
    _i=(_c+_f)/3
    _j=(_d+_e)/3
    return _h,_i,_j
def obf_sir(b, g, S0, I0, R0, t):
    _k,_l,_m=S0,I0,R0
    _n=[]
    _o=[]
    for _p in t:
        _q,_r,_s=_xyz(_k,_l,_m,b,g,_p)
        _n.append(_q)
        _o.append(_r)
        _m=_s
    return t,_n,_o

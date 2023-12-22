import numpy as _n, matplotlib.pyplot as _p
def _f(_i,_p,_r,_t):
  _v=lambda _a,_b: (-(1/_a)*_b,_r*_b,_b/_a-_r*_b)
  _y=_n.zeros((_t,3))
  _y[0]=_i
  for _j in range(_t-1):
    _k1=_v(*_y[_j])
    _k2=_v(*(_y[_j]+0.5*_k1))
    _k3=_v(*(_y[_j]-_k1+2*_k2))
    _y[_j+1]=_y[_j]+(_k1+4*_k2+_k3)/6
  return _y
_i=[10**6-1,1,0]
_p=14
_r=1/14
_t=100
_y=_f(_i,_p,_r,_t)
_p.figure()
_p.plot(_y)
_p.legend(['S','I','R'])
_p.show()

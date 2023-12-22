import numpy as np;p=np.dot;import matplotlib.pyplot as plt;plt.style.use('ggplot');j=0.5;k=0.03;n=100;ts=100;s=i=1.0/n;r=0-1.0/n;dt=ts/n;X=0+t*dt;S=[s];I=[i];R=[r];m=lambda t,s,i,r:[-j*s*i,j*s*i-k*r,k*r]
def rk(Y,t,dt,m):k1=m(t,*Y);k2=m(t+dt/2,*[y+dt*k1[c]/2 for c,y in enumerate(Y)]);k3=m(t+dt,*[y+dt*(-k1[c]+2*k2[c]) for c,y in enumerate(Y)]);return[y+dt*(k1[c]/6+2*k2[c]/3+k3[c]/6) for c,y in enumerate(Y)]
def f_m(t,s,i,r):return[-j*s*i,j*s*i-k*r,k*r]  
for t in np.arange(dt, ts+dt, dt):s,i,r=rk([s,i,r],t-dt,dt,f_m);S.append(s);I.append(i);R.append(r);X=np.append(X,t)
plt.plot(X,S,label='S');plt.plot(X,I,label='I');plt.plot(X,R,label='R');plt.legend();plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

L = 1.0
T = 50.0
dx = 1.0/50
dt = 1.0/1000
c = 1.5
x = np.arange(-L,L+dx,dx)
u = 2*x**2-c
diff = 0.002
I = np.zeros_like(x)
I[int(0.5/dx)] = 1.0
I[int(1.5/dx)] = 1.0

# I = np.abs(np.sin(x*2*np.pi))
fig,ax = plt.subplots()
ax.axis([-L,L,-2,4])
line1, = ax.plot(x,u)
line2, = ax.plot(x,I)
sols=[]

def speed(I, dx):
    didx = dudx(I,dx)

    return 1.0/(1+1000*np.abs(didx))

def dudx(u,dx):
    v = u.copy()
    v[1:-1] = 1.0/(2*dx)*(u[2:]-u[:-2])
    v[0] = 1.0/(2*dx)*(u[1]-u[-1])
    v[-1] = 1.0/(2*dx)*(u[0]-u[-2])
    return v.copy()

def d2udx2(u,dx):
    v = u.copy()
    v[1:-1] = 1.0/(dx**2)*(u[2:]-2*u[1:-1]+u[:-2])
    v[0] = 1.0/(dx**2)*(u[1]-2*u[0]+u[-1])
    v[-1] = 1.0/(dx**2)*(u[0]-2*u[-1]+u[-2])
    return v.copy()

for t in range(int(T/dt)):
    u = u + dt*speed(I,dx)*np.abs(dudx(u,dx)) + dt*diff*d2udx2(u,dx)
    sols.append(u.copy())

def animate(i):
    line1.set_data(x,sols[i])
    line2.set_data(x,I)
    return [line1,line2]

ani = animation.FuncAnimation(fig,animate,np.arange(0,len(sols)-1),interval=dt/10)
plt.show()

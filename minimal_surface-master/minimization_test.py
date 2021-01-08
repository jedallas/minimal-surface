from importlib import reload
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# disable unused import warnings for casadi
from casadi import *  # pylint: disable=W0614
import numpy as np
import minsurf as ms  # pylint: disable=E0401


m = 19
n = 19
X = np.linspace(0, 1, m)
Y = X


[X, Y] = np.meshgrid(X, Y)

sfc, sym_arg = ms.object_fuction(m,n)
# create a casadi function object from the symbolic expressions
s_f = Function('s_f', [sym_arg], [sfc])





# the initial value
Z = np.sin(np.ones((m,n))+0.001)
print("s_f(Z): ", s_f(Z))


z_0 = Z
z = z_0


# use a quick and dirty gradient descent method with
# an Armijo line search to minimize the surface
# functional

# set the Armijo line search and gradient descent
# parameters and tolerance

gamma = 1
delta = 0.1

alpha = 0.03  # learning rate
eps = 0.9


tol = 1e-5

norm_gradfz = tol + 1

it = 0
cnt = 0

v=np.zeros((m,n))
function_value=[]
norm_of_gradient=[]
while it <= 5000 and norm_gradfz >= tol:
    sigma = gamma

    # compute the gradient of the surface functional
    sym_gradf = gradient(sfc, sym_arg)
    sym_hessf=hessian(sfc,sym_arg)
    
    func_gradf = Function('func_gradf', [sym_arg], [sym_gradf])

    # compute the function  z (yielding a symbolic expression)
    fz = s_f(z)
    # convert the expression into a number
    fz = fz.toarray()

    # do the same with the gradient
    gradfz = func_gradf(z)
    gradfz = gradfz.toarray()

    # reshape the gradient to apply the standard
    # euclidean norm
    gradfz_reshaped = gradfz.T.reshape((m*n, 1))
    gradfz_sq = np.dot(gradfz_reshaped.T, gradfz_reshaped)
    norm_gradfz = np.sqrt(gradfz_sq)

    # perform the line search to determine the step size
    '''
    sigma = gamma
    phi = s_f(z - sigma * gradfz)
    phi = phi.toarray()
    while phi > (fz - delta * sigma * gradfz_sq):
        sigma = sigma/2
        phi = s_f(z - sigma * gradfz)
        phi = phi.toarray()
    
    # perform the gradient step
    z = z - sigma * gradfz
    '''
    #Momentun
    v = eps * v - alpha* gradfz  # 在这里进行速度更新
    z = z + v  # 使用动量来更新参数
    
    


    # set up the function values and the gradient
    # for the next iteration
    gradfz = func_gradf(z)
    gradfz = gradfz.toarray()

    gradfz_reshaped = gradfz.T.reshape((m*n, 1))
    gradfz_sq = np.dot(gradfz_reshaped.T, gradfz_reshaped)
    norm_gradfz = np.sqrt(gradfz_sq)
    fz = s_f(z)
    fz = fz.toarray()
    it = it + 1 
    function_value.append(fz[0][0])
    norm_of_gradient.append(norm_gradfz[0][0])

    # take a surface plot snapshot of every 10th iterate
    if(it % 10 == 0):
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #ax.set_zlim(0, 1)
        #ax.plot_surface(X, Y, z, color='c')
        #plt.savefig('./plot_temp/surface_test'+ str(cnt) +'.png')
        #plt.close(fig)
        #cnt = cnt + 1
        print("it, norm_gradfz: ", it, norm_gradfz)

# make a surface plot of the final iterate
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_zlim(0, 1)
ax.plot_surface(X, Y, z, color='c')
plt.savefig('./plot_temp/surface_test'+ str(it) +'.png')
plt.close(fig)


plt.title('Relative Objective Function Gap')

plt.plot(range(len(r2)),r2,label='1 + sin(2πx)')
plt.plot(range(len(r3)),r3,label='1 + cos(1/(x + 0.001))')
plt.plot(range(len(r1)),r1,label='1/2-|y-1/2|')
plt.plot(range(len(r4)),r4,label='1/1+exp(xy)')
plt.plot(range(len(r5)),r5,label='1 + arcsin(−1 + 2√xy)')
plt.legend()


plt.title('Norm of Gradients')
plt.plot(range(len(r2_norm)),r2_norm,label='1 + sin(2πx)')
plt.plot(range(len(r3_norm)),r3_norm,label='1 + cos(1/(x + 0.001))')
plt.plot(range(len(r1_norm)),r1_norm,label='1/2-|y-1/2|')
plt.plot(range(len(r4_norm)),r4_norm,label='1/1+exp(xy)')
plt.plot(range(len(r5_norm)),r5_norm,label='1 + arcsin(−1 + 2√xy)')
plt.legend()



plt

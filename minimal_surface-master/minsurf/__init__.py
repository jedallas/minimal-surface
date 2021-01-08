import numpy as np
import scipy.integrate
# disable unused import warnings for casadi
from casadi import *  # pylint: disable=W0614

name = "minsurf"


def function(x,y):
    
    #z = 1 + np.sin(2*np.pi*x)
    #z = 1 + np.cos(1/(x+0.001))
    z = 0.5 - abs(y-0.5)
    #z = 1/(1 + np.exp(x*y))
    #z = 1 + np.arcsin(-1+2*np.sqrt(x*y))
    return z

def object_fuction(m,n):
    a = np.linspace(0, 1, m)
    b = np.linspace(0, 1, n)
    dic={}
    X=SX.sym('X',m,n)
    for i in range(m):
        for j in range(n):
             x=SX.sym('x_{0}_{1}'.format(i,j), 1)
             dic[i,j]=np.array([a[i],b[j],x])
             X[i,j]=x
             
             
    obj_fuc = 0;
    for i in range(m-1):
        for j in range(n-1):
            lt=cross(dic[i+1,j] - dic[i,j],dic[i,j+1] - dic[i,j])
            lt = sqrt(dot(lt,lt))/2
            ut=cross(dic[i+1,j] - dic[i+1,j+1],dic[i,j+1] - dic[i+1,j+1])
            ut = sqrt(dot(ut,ut))/2
        
            obj_fuc = obj_fuc + lt + ut
    
    for i in range(m):
        for j in range(n):
            
            #print(z)
            _,__,x=dic[i,j]
            z=function(_,__)
            obj_fuc=obj_fuc+dot(z-x,z-x)
            
    return obj_fuc,X

def surface_composite_newton_cotes(left_bd, right_bd, n=20, m=2):
    """Assemble a casadi-symbolic composite Newton-Cotes
    surface integration formula for a function z(x,y)
    assumed to be discretized on the grid 
    [left_bd, right_bd]^2 with N = m + (n-1)*(m-1)
    equally spaced points in every direction.


    Arguments:
        n {[int]} -- [number of individual interval 
        partitions of [left_bd, right_bd] into intervals
        of m points each]
        m {[int]} -- [number of points within each partial
        interval of the composite Newton-Cotes procedure]
        left_bd {[int]} -- [lower boundary of the discretization
        interval in each dimension]
        right_bd {[int]} -- [lower boundary of the discretization
        interval in each dimension]

    Returns
        sfc {[SX]} -- [casadi symbolic expression of the
        resulting Newton-Cotes surface integral approximation
        where the input argument is pased as an NxN SX 
        symbolic expression]
        z {[SX]} -- [casadi symbolic expression of the
        input grid z]
    """
    # actual number of discrete points in every dimension
    N = m + (n-1)*(m-1)
    z = SX.sym('z_', (N, N))

    # distance between two discretization points
    h = (right_bd - left_bd)/(N-1)
    

    # generate Newton-Cotes coefficients for each partial interval
    print("Generating Newton-Cotes coeffs...")
    [c, B] = scipy.integrate.newton_cotes(m-1)
    c = 1/(m-1) * c
    print("...done!\n")
    # define the approximated surface function
    # initiate with zero for iterated assembly
    sfc = SX.sym('sfc', 1)
    sfc[0] = 0

    # write Newton-Cotes coefficients into symbolic vector
    coeff = SX(c)
    c = coeff

    print("Assembling surface functional...")
    s = 0

    for k in range(0, n):
        for l in range(0, n):
            for i in range(0, m):
                for j in range(0, m):

                    ind_i = k*(m-1) + i
                    ind_j = l*(m-1) + j

                    if(ind_i == N - 1):
                        dy = (z[ind_i - 1, ind_j]-z[ind_i, ind_j])/h
                    else:
                        dy = (z[ind_i + 1, ind_j]-z[ind_i, ind_j])/h

                    if(ind_j == N - 1):
                        dx = (z[ind_i, ind_j - 1]-z[ind_i, ind_j])/h
                    else:
                        dx = (z[ind_i, ind_j + 1]-z[ind_i, ind_j])/h


                    sfc = sfc + c[i]*c[j]*sqrt(1 + dx**2 + dy**2)
                    s = s + 1

    sfc = (((right_bd - left_bd)/(n))**2) * sfc
    print("...done! Performed s = ", s, "assembly operations.\n")
    return sfc,z,N


def interior_constrained_surface_composite_newton_cotes(left_bd, right_bd, B, n=20, m=2, interior = "arch"):
    """Assemble a casadi-symbolic composite Newton-Cotes
    surface integration formula for a function z(x,y)
    assumed to be discretized on the grid 
    [left_bd, right_bd]^2 with N = m + (n-1)*(m-1)
    equally spaced points in every direction.


    Arguments:
        n {[int]} -- [number of individual interval 
        partitions of [left_bd, right_bd] into intervals
        of m points each]
        m {[int]} -- [number of points within each partial
        interval of the composite Newton-Cotes procedure]
        left_bd {[int]} -- [lower boundary of the discretization
        interval in each dimension]
        right_bd {[int]} -- [lower boundary of the discretization
        interval in each dimension]
        interior {[string]} -- [selects which constraints to 
        apply in the interior of the rectangle; interior = "arch" for 
        a quadratic function constraint on a line segment
        in the interior and interior = "peaks" for two constant values]

    Returns
        sfc {[SX]} -- [casadi symbolic expression of the
        resulting Newton-Cotes surface integral approximation
        where the input argument is pased as an NxN SX 
        symbolic expression]
        z {[SX]} -- [casadi symbolic expression of the
        input grid z]
    """
    # actual number of discrete points in every dimension
    N = m + (n-1)*(m-1)
    z_complete = SX.sym('z', N, N)

    # directly altering z_complete repeatedly and 
    # inconsistently threw type errors for some reason
    # this fix prevents it though it shouldn't be
    # necessary 
    z_inter = reshape(z_complete, (N**2,1))
    z_inter = z_complete
    z_inter = reshape(z_inter, (N,N))

    # apply boundary constraints given by B
    '''
    z_inter[0,:] = B[:,0]
    z_inter[:,0] = B[:,1]
    z_inter[-1,:] = B[:,2]
    z_inter[:,-1] = B[:,3]
    '''

    if interior == "arch":
        # interior arch constraint
        b = np.linspace(0,1,3*N//4 - N//4)
        b = -(b - 0.5)**2 + 1
        z_inter[N//4:3*N//4,N//2] = b
    elif interior == "peaks":
        # interior two peaks constraint
        z_inter[N//4, N//2] = 1
        z_inter[3*N//4, N//2] = 1
    else:
        print("Invalid interior constraint setting!\nApplying no interior constraints.")
    

    z = z_inter

    # distance between two discretization points
    h = (right_bd - left_bd)/(N-1)

    # generate Newton-Cotes coefficients for each partial interval
    print("Generating Newton-Cotes coeffs...")
    [c, _] = scipy.integrate.newton_cotes(m-1)
    c = 1/(m-1) * c
    print("...done!\n")
    # define the approximated surface function
    # initiate with zero for iterated assembly
    sfc = SX.sym('sfc', 1)
    sfc[0] = 0

    # write Newton-Cotes coefficients into symbolic vector
    coeff = SX(c)
    c = coeff

    print("Assembling surface functional...")
    s = 0

    for k in range(0, n):
        for l in range(0, n):
            for i in range(0, m):
                for j in range(0, m):

                    ind_i = k*(m-1) + i
                    ind_j = l*(m-1) + j

                    if(ind_i == N - 1):
                        dy = (z[ind_i - 1, ind_j]-z[ind_i, ind_j])/h
                    else:
                        dy = (z[ind_i + 1, ind_j]-z[ind_i, ind_j])/h

                    if(ind_j == N - 1):
                        dx = (z[ind_i, ind_j - 1]-z[ind_i, ind_j])/h
                    else:
                        dx = (z[ind_i, ind_j + 1]-z[ind_i, ind_j])/h
             
                    sfc = sfc + c[i]*c[j]*sqrt(1 + dx**2 + dy**2)
                    s = s + 1

    sfc = (((right_bd - left_bd)/(n))**2) * sfc
    print("...done! Performed s = ", s, "assembly operations.\n")
    return sfc,z_complete,N


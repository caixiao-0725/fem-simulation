import warp as wp


@wp.kernel
def addVec3(x:wp.array(dtype=wp.vec3),y:wp.array(dtype=wp.vec3)):
    idx = wp.tid()
    x[idx] = x[idx]+ y[idx]

@wp.kernel
def subVec3(x:wp.array(dtype=wp.vec3),y:wp.array(dtype=wp.vec3)):
    idx = wp.tid()
    # if y[idx] == wp.vec3f():
    #     print(idx)
    x[idx] = x[idx]- y[idx]

@wp.kernel
def update_x(x:wp.array(dtype=wp.vec3),grad:wp.array(dtype=wp.vec3),dt:wp.float32):
    idx = wp.tid()
    x[idx] -= grad[idx]*dt

@wp.kernel
def update_deltaX_kernel(x:wp.array(dtype=wp.vec3),deltaX:wp.array(dtype=wp.vec3),index2vertex:wp.array(dtype=wp.int32)):
    idx = wp.tid()
    i = index2vertex[idx]
    x[i] =x[i] - deltaX[idx]

@wp.kernel
def minues_grad(x:wp.array(dtype=wp.vec3),x_res:wp.array(dtype=wp.vec3),grad:wp.array(dtype=wp.vec3),dt:wp.float32):
    idx = wp.tid()
    x_res[idx] = x[idx]-grad[idx]*dt

@wp.kernel
def square_sum(x:wp.array(dtype=wp.vec3),res:wp.array(dtype=wp.float32)):
    idx = wp.tid()
    temp_res = wp.dot(x[idx],x[idx])
    wp.atomic_add(res,0,temp_res)

#use conjugate gradient to solve Ax=b (A:3x3  b:3x1  x:3x1)
@wp.func
def solve3x3(A:wp.mat33f,b:wp.vec3f,x:wp.vec3f):
    r = wp.vec3f()
    p = wp.vec3f()
    Ap = wp.vec3f()
    old_r_norm = 0.0
    r_norm = 0.0
    dot = 0.0
    alpha = 0.0
    beta = 0.0
    r = b
    r_norm = wp.dot(r,r)
    if r_norm < 1e-6:
        return x
    p = r
    for l in range(3):
        for i in range(3):
            Ap[i] = 0.0
        Ap = wp.mul(A,p)
        dot = wp.dot(p,Ap)
        if dot<0:
            print('not spd')
        if dot < 1e-6:
            return x
        alpha = r_norm/dot
        x = x+alpha*p
        r = r-alpha*Ap
        old_r_norm = r_norm
        r_norm = wp.dot(r,r)
        if r_norm < 1e-6:
            return x
        beta = r_norm/old_r_norm
        p = r+beta*p


@wp.kernel
def jacobi_iteration_offset(x:wp.array(dtype=wp.vec3f),value:wp.array(dtype=wp.mat33f),diag_offset:wp.array(dtype=wp.int32),b:wp.array(dtype=wp.vec3f)):
    idx = wp.tid()
    diag = value[diag_offset[idx]]
    # if idx == 0:
    #     print(value[diag_offset[idx]])
    #     print(b[idx])
    #     print(x[idx])
    #     print(wp.mul(wp.inverse(diag),b[idx]))
    x[idx] = solve3x3(diag,b[idx],x[idx])
    # if idx == 0:
    #     print(x[idx])

@wp.kernel
def jacobi_iteration(x:wp.array(dtype=wp.vec3f),value:wp.array(dtype=wp.mat33f),b:wp.array(dtype=wp.vec3f)):
    idx = wp.tid()
    diag = value[idx]
    x[idx] = solve3x3(diag,b[idx],x[idx])
    # if idx == 0:
    #     print(diag)

@wp.kernel
def spd_matrix33f(x:wp.array(dtype=wp.mat33f)):
    idx = wp.tid()
    A = x[idx]
    V = wp.mat33f()
    D = wp.vec3f()
    wp.eig3(A,V,D)
    for i in range(3):
        if D[i] < 0:
            D[i] = 0.001
    x[idx] = wp.mul(V,wp.mul(wp.diag(D),wp.transpose(V)))

@wp.kernel
def Colored_GS_MF_Kernel(x:wp.array(dtype=wp.vec3f),value:wp.array(dtype=wp.mat33f),b:wp.array(dtype=wp.vec3f),base:wp.int32,number:wp.int32):
    idx = wp.tid()
    if idx>=number: 
        return
    t = base+idx
    x[t] = solve3x3(value[t],b[t],x[t])


@wp.kernel()
def compute_partial_elastic_energy_X(x:wp.array(dtype=wp.vec3),hexagons:wp.array(dtype=wp.int32,ndim=2),vertex2index:wp.array(dtype=wp.int32),
                shapeFuncGrad:wp.array(dtype=wp.float32,ndim=3),det_pX_peps:wp.array(dtype=wp.float32,ndim=2),inverse_pX_peps:wp.array(dtype=wp.mat33f,ndim=2),
                IM:wp.array(dtype=wp.mat33f),LameMu:wp.array(dtype=wp.float32),LameLa:wp.array(dtype=wp.float32),
                grad:wp.array(dtype=wp.vec3)):
    idx = wp.tid()
    whichQuadrature = idx%8
    hex = idx//8
    F = wp.mat33f()
    for row in range(3):
        for col in range(3):
            value_now = 0.0
            for i in range(8):
                value_now += x[hexagons[hex][i]][row]*shapeFuncGrad[i][whichQuadrature][col]
            F[row,col] = value_now
    F = F @ inverse_pX_peps[hex][whichQuadrature]
    
    E = 0.5*(wp.transpose(F)@F-IM[0])
    P = F@(2.0*LameMu[0]*E+LameLa[0]*wp.trace(E)*IM[0])
    for i in range(8):
        shapeFuncGradNow = wp.vec3f()
        for j in range(3):
            shapeFuncGradNow[j] = shapeFuncGrad[i][whichQuadrature][j]
        temAns = wp.mul(P@wp.transpose(inverse_pX_peps[hex][whichQuadrature]),shapeFuncGradNow)*det_pX_peps[hex][whichQuadrature]
        wp.atomic_add(grad,vertex2index[hexagons[hex][i]],temAns)
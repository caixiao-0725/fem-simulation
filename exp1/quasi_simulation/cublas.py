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
def minues_grad(x:wp.array(dtype=wp.vec3),x_res:wp.array(dtype=wp.vec3),grad:wp.array(dtype=wp.vec3),dt:wp.float32):
    idx = wp.tid()
    x_res[idx] = x[idx]-grad[idx]*dt

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
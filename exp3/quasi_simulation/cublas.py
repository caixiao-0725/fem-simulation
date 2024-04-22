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
    x[i] =x[i] + deltaX[idx]

@wp.kernel
def update_deltaX_kernel_ordered(x:wp.array(dtype=wp.vec3),deltaX:wp.array(dtype=wp.vec3),vertex2index:wp.array(dtype=wp.int32)):
    idx = wp.tid()
    i = vertex2index[idx]
    x[i] =x[i] + deltaX[idx]

@wp.kernel
def add_grad(x:wp.array(dtype=wp.vec3),x_res:wp.array(dtype=wp.vec3),grad:wp.array(dtype=wp.vec3),dt:wp.float32):
    idx = wp.tid()
    x_res[idx] = x[idx]+grad[idx]*dt

@wp.kernel
def square_sum(x:wp.array(dtype=wp.vec3),res:wp.array(dtype=wp.float32)):
    idx = wp.tid()
    temp_res = wp.dot(x[idx],x[idx])
    wp.atomic_add(res,0,temp_res)

@wp.kernel
def Inf_norm(x:wp.array(dtype=wp.vec3),res:wp.array(dtype=wp.float32),index2vertex:wp.array(dtype=wp.int32),pinned:wp.array(dtype=wp.int32)):
    idx = wp.tid()
    # if pinned[index2vertex[idx]] == 1:
    #     return
    temp_x = x[idx]
    temp_max = wp.abs(temp_x[2])
    for i in range(2):
        if wp.abs(temp_x[i])>temp_max:
            temp_max = wp.abs(temp_x[i])
    wp.atomic_max(res,0,temp_max)
    


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
    if r_norm < 1e-10:
        return x
    p = r
    for l in range(3):
        for i in range(3):
            Ap[i] = 0.0
        Ap = wp.mul(A,p)
        dot = wp.dot(p,Ap)
        if dot<0:
            print('not spd')
        if dot < 1e-10:
            return x
        alpha = r_norm/dot
        x = x+alpha*p
        r = r-alpha*Ap
        old_r_norm = r_norm
        r_norm = wp.dot(r,r)
        if r_norm < 1e-10:
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
def jacobi_iteration(x:wp.array(dtype=wp.vec3f),value:wp.array(dtype=wp.mat33f),b:wp.array(dtype=wp.vec3f),offset:int):
    idx = wp.tid()
    diag = value[idx+offset]
    x[idx] = solve3x3(diag,b[idx],x[idx])
    # if idx == 0:
    #      print(diag)

@wp.kernel
def spd_matrix33f(x:wp.array(dtype=wp.mat33f),value:float):
    idx = wp.tid()
    A = x[idx]
    V = wp.mat33f()
    D = wp.vec3f()
    wp.eig3(A,V,D)
    for i in range(3):
        if D[i] < 0:
            D[i] = value
    x[idx] = wp.mul(V,wp.mul(wp.diag(D),wp.transpose(V)))
    # if idx<3:
    #     print(x[idx])

@wp.kernel
def Colored_GS_MF_Kernel(x:wp.array(dtype=wp.vec3f),value:wp.array(dtype=wp.mat33f),b:wp.array(dtype=wp.vec3f),base:wp.int32,number:wp.int32,diag_offset:int):
    idx = wp.tid()
    if idx>=number: 
        return
    t = base+idx
    x[t] = solve3x3(value[t+diag_offset],b[t],x[t])

@wp.kernel
def block_values_reorder(A_init:wp.array(dtype=wp.mat33f),A:wp.array(dtype=wp.mat33f),index:wp.array(dtype=wp.int32)):
    idx = wp.tid()
    i = index[idx]
    A[idx] = A_init[i]

@wp.kernel
def scal(x:wp.array(dtype=wp.vec3f),a:wp.float32):
    idx = wp.tid()
    x[idx] = x[idx]*a

@wp.kernel
def scal_(x:wp.array(dtype=wp.vec3f),y:wp.array(dtype=wp.vec3f),a:wp.float32):
    idx = wp.tid()   
    y[idx] = x[idx]*a


@wp.kernel
def axpy(y:wp.array(dtype=wp.vec3f),x:wp.array(dtype=wp.vec3f),a:wp.float32):
    idx = wp.tid()
    y[idx] = y[idx]+a*x[idx]

@wp.kernel
def axpby(x:wp.array(dtype=wp.vec3f),y:wp.array(dtype=wp.vec3f),a:wp.float32,b:float):
    idx = wp.tid()
    y[idx] = b*y[idx]+a*x[idx]

@wp.kernel
def z_axpby(z:wp.array(dtype=wp.vec3f),x:wp.array(dtype=wp.vec3f),y:wp.array(dtype=wp.vec3f),a:wp.float32,b:float):
    idx = wp.tid()
    z[idx] = a*x[idx]+b*y[idx]    

'''
下面的函数是adam用的
'''
@wp.kernel
def updateM(x:wp.array(dtype=wp.vec3f),y:wp.array(dtype=wp.vec3f),a:wp.float32,b:wp.float32):
    idx = wp.tid()
    y[idx] = a*x[idx]+b*y[idx]


@wp.kernel
def updateV(x:wp.array(dtype=wp.vec3f),y:wp.array(dtype=wp.vec3f),a:wp.float32,b:wp.float32):
    idx = wp.tid()
    y[idx] = a*wp.cw_mul(x[idx],x[idx])+b*y[idx]



@wp.kernel
def updateX(x:wp.array(dtype=wp.vec3f),m:wp.array(dtype=wp.vec3f),v:wp.array(dtype=wp.vec3f),lr:wp.float32,epsilon:wp.float32):
    idx = wp.tid()
    temp_v = v[idx]
    temp_m = m[idx]
    ans = wp.vec3f()

    for i in range(3):
        ans[i] = temp_m[i]/(wp.sqrt(temp_v[i])+epsilon)
    x[idx] -= lr*ans
    
    

@wp.kernel
def cublasSdot(x:wp.array(dtype=wp.vec3f),y:wp.array(dtype=wp.vec3f),res:wp.array(dtype=wp.float32)):
    idx = wp.tid()
    wp.atomic_add(res,0,wp.dot(x[idx],y[idx]))

@wp.kernel
def V2I(x:wp.array(dtype=wp.vec3f),x_gpu:wp.array(dtype=wp.vec3f),vertex2index:wp.array(dtype=wp.int32)):
    idx = wp.tid()
    i = vertex2index[idx]
    x[i] = x_gpu[idx]

@wp.kernel
def I2V(x:wp.array(dtype=wp.vec3f),x_gpu:wp.array(dtype=wp.vec3f),index2vertex:wp.array(dtype=wp.int32)):
    idx = wp.tid()
    i = index2vertex[idx]
    x[i] = x_gpu[idx]

@wp.kernel
def Sum_8(res:wp.array(dtype=wp.float32),v:wp.array(dtype=wp.float32,ndim=2)):
    idx = wp.tid()
    res[idx] = v[idx,0]+v[idx,1]+v[idx,2]+v[idx,3]+v[idx,4]+v[idx,5]+v[idx,6]+v[idx,7]

@wp.kernel
def updateFaceNorm(x:wp.array(dtype=wp.vec3f),f:wp.array(dtype=wp.vec3i),norm:wp.array(dtype=wp.vec3f)):
    idx = wp.tid()
    p0 = f[idx][0]
    p1 = f[idx][1]
    p2 = f[idx][2]
    norm[idx] = wp.normalize(wp.cross(x[p1]-x[p0],x[p2]-x[p0]))

@wp.kernel
def updateVertNorm(x:wp.array(dtype=wp.vec3f),f:wp.array(dtype=wp.vec3i),norm:wp.array(dtype=wp.vec3f)):
    idx = wp.tid()
    for i in range(3):
        wp.atomic_add(x,f[idx][i],norm[idx])

@wp.kernel
def compute_fix_hessian(vertex2index:wp.array(dtype=wp.int32),fix:wp.array(dtype=wp.int32),control_mag:float,offset:int,hessian:wp.array(dtype=wp.mat33f),fix_idx:wp.array(dtype=wp.int32,ndim=2),fix_value:wp.array(dtype=wp.float32,ndim=2)):
    idx = wp.tid()
    IM = wp.mat33f(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
    if fix[idx]==1:
        for j in range(8):
            if fix_idx[idx][j] != -1:
                jdx = fix_idx[idx][j]
                wp.atomic_add(hessian,offset+vertex2index[jdx],fix_value[idx][j]*fix_value[idx][j]*control_mag*IM)

@wp.kernel
def print_hessian(hessian:wp.array(dtype=wp.mat33f)):
    idx = wp.tid()
    print(hessian[idx])
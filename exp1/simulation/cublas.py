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
def updateVelocity(x:wp.array(dtype=wp.vec3f),x_old:wp.array(dtype=wp.vec3f),v:wp.array(dtype=wp.vec3f),inv_t:wp.float32):
    idx = wp.tid()
    v[idx] = (x[idx]-x_old[idx])*inv_t

@wp.kernel
def selectKernel(p0:wp.vec3f,dir:wp.vec3f,x:wp.array(dtype=wp.vec3f),face:wp.array(dtype=wp.vec3i),select_num:wp.array(dtype=wp.int32),select_face:wp.array(dtype=wp.int32),select_distance:wp.array(dtype=wp.float32)):
    idx = wp.tid()
    f0 = face[idx][0]
    f1 = face[idx][1]
    f2 = face[idx][2]
    x0 = x[f0]
    x1 = x[f1]
    x2 = x[f2]
    e1 = x1-x0
    e2 = x2-x0
    s1 = wp.cross(dir,e2)
    divisor = wp.dot(s1,e1)
    if divisor == 0:
        return 
    tt = p0-x0
    b1 = wp.dot(tt,s1)
    if divisor>0 and (b1<0 or b1>divisor) :
        return
    if divisor<0 and (b1>0 or b1<divisor):
        return
    s2 = wp.cross(tt,e1)
    b2 = wp.dot(dir,s2)
    if divisor>0 and (b2<0 or b1+b2>divisor):
        return
    if divisor<0 and (b2>0 or b1+b2<divisor):
        return
    t = wp.dot(e2,s2)/divisor
    if t<0:
        return
    id = wp.atomic_add(select_num,0,1)
    select_face[id] = idx
    select_distance[id] = t
    return

@wp.kernel
def Control_Kernel(select_v:int,more_fixed_gpu:wp.array(dtype=wp.int32),x:wp.array(dtype=wp.vec3f)):
    idx = wp.tid()
    more_fixed_gpu[idx] =0
    dist2 = wp.dot(x[idx]-x[select_v],x[idx]-x[select_v])
    if dist2<0.002:
        more_fixed_gpu[idx] = 1

@wp.kernel
def Fixed_Update_Kernel(x:wp.array(dtype=wp.vec3f),fixed_x:wp.array(dtype=wp.vec3f),more_fixed_gpu:wp.array(dtype=wp.int32),pin:wp.array(dtype=wp.int32),dir:wp.vec3f):
    idx = wp.tid()
    if more_fixed_gpu[idx] == 1 and pin[idx] == 0:
        fixed_x[idx] = x[idx] + dir

@wp.kernel
def Hessian_Diag_Kernel(values:wp.array(dtype=wp.mat33f),offset:wp.int32,vertex2index:wp.array(dtype=wp.int32),more_fixed_gpu:wp.array(dtype=wp.int32),pin:wp.array(dtype=wp.int32),control_mag:float):
    idx = wp.tid()
    if more_fixed_gpu[idx] == 1 or pin[idx] == 1:
        i = vertex2index[idx]  
        values[offset+i] = control_mag* wp.mat33f(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
@wp.kernel
def compute_partial_more_fixed_energy_X(x:wp.array(dtype=wp.vec3f),vertex2index:wp.array(dtype=wp.int32),more_fixed_gpu:wp.array(dtype=wp.int32),grad:wp.array(dtype=wp.vec3f),fixed_x:wp.array(dtype=wp.vec3f),control_mag:float):
    idx = wp.tid()
    if more_fixed_gpu[idx] == 1:
        i = vertex2index[idx]
        grad[i] += control_mag * (fixed_x[idx]-x[idx])
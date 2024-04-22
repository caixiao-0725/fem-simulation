import warp as wp



@wp.kernel()
def compute_rest_length(x:wp.array(dtype=wp.vec3),edge:wp.array(dtype=wp.vec2i),rest_length:wp.array(dtype=wp.float32)):
    idx = wp.tid()
    p0 = edge[idx][0]
    p1 = edge[idx][1]
    rest_length[idx] = wp.length(x[p0]-x[p1])


@wp.kernel()
def compute_elastic_energy(x:wp.array(dtype=wp.vec3f),edge:wp.array(dtype=wp.vec2i),rest_length:wp.array(dtype=wp.float32),k:float,
                            loss:wp.array(dtype=wp.float32)):
    idx = wp.tid()
    p0 = edge[idx][0]
    p1 = edge[idx][1]
    delta_x = x[p0]-x[p1]
    delta_l = wp.length(delta_x)-rest_length[idx]
    energy = 0.5*k*delta_l*delta_l
    wp.atomic_add(loss,0,energy) 

@wp.kernel
def compute_elastic_gradient(x:wp.array(dtype=wp.vec3f),edge:wp.array(dtype=wp.vec2i),rest_length:wp.array(dtype=wp.float32),k:float,
                             gradient:wp.array(dtype=wp.vec3f)):
    idx = wp.tid()
    p0 = edge[idx][0]
    p1 = edge[idx][1]
    delta_x = x[p0]-x[p1]
    delta_l = k*(rest_length[idx]/wp.length(delta_x)-1.0)
    grad = delta_l*delta_x
    wp.atomic_add(gradient,p0,grad)
    wp.atomic_add(gradient,p1,-grad)

@wp.kernel
def compute_Inertia_Gradient(grad:wp.array(dtype=wp.vec3f),x_inertia:wp.array(dtype=wp.vec3f),x:wp.array(dtype=wp.vec3f),m:wp.array(dtype=wp.float32),inv_t:wp.float32):
    idx = wp.tid()
    c = m[idx]*inv_t*inv_t
    grad[idx] -= c*(x_inertia[idx]-x[idx])

@wp.kernel()
def compute_gravity_Gradient(m:wp.array(dtype=wp.float32),g:float,grad:wp.array(dtype=wp.vec3)):
    idx = wp.tid()
    grad[idx][1] += m[idx]*g

@wp.kernel
def compute_elastic_hessian(x:wp.array(dtype=wp.vec3f),edge:wp.array(dtype=wp.vec2i),rest_length:wp.array(dtype=wp.float32),k:float,
                            hessian:wp.array(dtype=wp.mat33f),offset:wp.array(dtype=wp.int32)):
    idx = wp.tid()
    len = rest_length[idx]
    p0 = edge[idx][0]
    p1 = edge[idx][1]
    d = x[p0]-x[p1]
    len_new = wp.length(d)
    a = k*len/len_new
    b = a/len_new/len_new
    I = wp.mat33f(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
    H = (k-a)*I + b*wp.outer(d,d)
    wp.atomic_add(hessian,offset[4*idx],H)
    wp.atomic_add(hessian,offset[4*idx+1],-H)
    wp.atomic_add(hessian,offset[4*idx+2],-H)
    wp.atomic_add(hessian,offset[4*idx+3],H)


@wp.kernel
def Hessian_Mass_Kernel(m:wp.array(dtype=wp.float32),Hessian:wp.array(dtype=wp.mat33f),offset:wp.array(dtype=wp.int32),inv_t:float):
    idx = wp.tid()
    Hessian[offset[idx]] += m[idx]*inv_t*inv_t*wp.mat33f(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)

@wp.kernel
def Basic_Update_Kernel(x:wp.array(dtype=wp.vec3f),v:wp.array(dtype=wp.vec3f),damping:float,dt:wp.float32):
    idx = wp.tid()
    v[idx] *=damping
    x[idx] += v[idx]*dt
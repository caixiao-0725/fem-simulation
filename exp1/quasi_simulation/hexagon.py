import warp as wp


@wp.kernel()
def prepare_kernal(x:wp.array(dtype=wp.vec3),hexagons:wp.array(dtype=wp.int32,ndim=2),shapeFuncGrad:wp.array(dtype=wp.float32,ndim=3),
                    det_pX_peps:wp.array(dtype=wp.float32,ndim=2),inverse_pX_peps:wp.array(dtype=wp.mat33f,ndim=2)):
    idx = wp.tid()
    hex = idx//8
    whichQuadrature = idx%8
    F = wp.mat33f()
    for row in range(3):
        for col in range(3):
            value_now = 0.0
            for i in range(8):
                value_now += x[hexagons[hex][i]][row]*shapeFuncGrad[i][whichQuadrature][col]
            F[row,col] = value_now
    det_pX_peps[hex][whichQuadrature] = wp.determinant(F)
    inverse_pX_peps[hex][whichQuadrature] = wp.inverse(F)

@wp.kernel()
def compute_elastic_energy(x:wp.array(dtype=wp.vec3),hexagons:wp.array(dtype=wp.int32,ndim=2),
                            shapeFuncGrad:wp.array(dtype=wp.float32,ndim=3), det_pX_peps:wp.array(dtype=wp.float32,ndim=2),inverse_pX_peps:wp.array(dtype=wp.mat33f,ndim=2),
                            IM:wp.array(dtype=wp.mat33f),LameMu:wp.array(dtype=wp.float32),LameLa:wp.array(dtype=wp.float32),
                            loss:wp.array(dtype=wp.float32)):
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

    sum_ = 0.0
    for row in range(3):
        for col in range(3):
            sum_ += E[row,col]*E[row,col]
    Psi = sum_*LameMu[0] + 0.5*LameLa[0]*wp.trace(E)@wp.trace(E)

    energy = Psi*det_pX_peps[hex][whichQuadrature]     

    wp.atomic_add(loss,0,energy) 

@wp.kernel()
def compute_gravity_energy(x:wp.array(dtype=wp.vec3),m:wp.array(dtype=wp.float32),g:wp.array(dtype=wp.float32),loss:wp.array(dtype=wp.float32)):
    idx = wp.tid()
    energy = -m[0]*g[0]*x[idx][1]
    wp.atomic_add(loss,0,energy)




@wp.kernel()
def compute_elastic_hessian(x:wp.array(dtype=wp.vec3),hexagons:wp.array(dtype=wp.int32,ndim=2),shapeFuncGrad:wp.array(dtype=wp.float32,ndim=3),
                   det_pX_peps:wp.array(dtype=wp.float32,ndim=2),inverse_pX_peps:wp.array(dtype=wp.mat33f,ndim=2),
                   IM:wp.array(dtype=wp.mat33f),LameMu:wp.array(dtype=wp.float32),LameLa:wp.array(dtype=wp.float32),
                   MF_value:wp.array(dtype=wp.mat33f),offset:wp.array(dtype=wp.int32)):
    idx = wp.tid()
    hex = idx//64
    idx_ = idx%64
    whichQuadrature = idx_//8
    whichShapeFunc = idx_%8
    F = wp.mat33f()
    for row in range(3):
        for col in range(3):
            value_now = 0.0
            for i in range(8):
                value_now += x[hexagons[hex][i]][row]*shapeFuncGrad[i][whichQuadrature][col]
            F[row,col] = value_now
    F = F @ inverse_pX_peps[hex][whichQuadrature]
    E = 0.5*(wp.transpose(F)@F-IM[0])

    shapeFuncGradNow = wp.vec3f()
    for j in range(3):
        shapeFuncGradNow[j] = shapeFuncGrad[whichShapeFunc][whichQuadrature][j]

    for ii in range(8):
        temAnsForHessian = wp.mat33f()
        # id0 = hexagons[hex][whichShapeFunc]
        # id1 = hexagons[hex][ii]
        for i in range(3):
            dF = wp.mat33f()
            for j in range(3):
                dF[i,j] = shapeFuncGrad[ii][whichQuadrature][0]*inverse_pX_peps[hex][whichQuadrature][0,j]+shapeFuncGrad[ii][whichQuadrature][1]*inverse_pX_peps[hex][whichQuadrature][1,j]+shapeFuncGrad[ii][whichQuadrature][2]*inverse_pX_peps[hex][whichQuadrature][2,j]
            dE = (wp.transpose(dF)@F+wp.transpose(F)@dF)*0.5
            dP = dF@(2.0*LameMu[0]*E+LameLa[0]*wp.trace(E)*IM[0])+F@(2.0*LameMu[0]*dE+LameLa[0]*wp.trace(dE)*IM[0])
            temAns = wp.mul(dP@(wp.transpose(inverse_pX_peps[hex][whichQuadrature])),shapeFuncGradNow)*det_pX_peps[hex][whichQuadrature]           
            for j in range(3):
                temAnsForHessian[j,i] = temAns[j]
        wp.atomic_add(MF_value,offset[hex*64+whichShapeFunc*8+ii],temAnsForHessian)
        

@wp.kernel()
def compute_partial_gravity_energy_X(m:wp.array(dtype=wp.float32),g:wp.array(dtype=wp.float32),grad:wp.array(dtype=wp.vec3)):
    idx = wp.tid()
    grad[idx][1] -= m[0]*g[0]

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

@wp.kernel()
def compute_partial_elastic_energy_X_noOrder(x:wp.array(dtype=wp.vec3),hexagons:wp.array(dtype=wp.int32,ndim=2),
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
        wp.atomic_add(grad,hexagons[hex][i],temAns)

@wp.kernel()
def pin(x:wp.array(dtype=wp.vec3),pin_pos:wp.array(dtype=wp.vec3),pin_list:wp.array(dtype=wp.int32)):
    idx = wp.tid()
    x[pin_list[idx]] = pin_pos[idx]

@wp.kernel()
def build_constraints(hexagons:wp.array(dtype=wp.int32,ndim=2),tet_update_offset:wp.array(dtype=wp.vec2i)):
    idx = wp.tid()
    for i in range(8):
        for j in range(8):
            tet_update_offset[idx*64+i*8+j][0] = hexagons[idx][i]
            tet_update_offset[idx*64+i*8+j][1] = hexagons[idx][j]


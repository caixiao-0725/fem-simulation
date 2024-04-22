import warp as wp
@wp.kernel()
def compute_mat_F(x:wp.array(dtype=wp.vec3f),hexagons:wp.array(dtype=wp.int32,ndim=2),shapeFuncGrad:wp.array(dtype=wp.vec3f,ndim=2),
                  det_pX_peps:wp.array(dtype=wp.float32,ndim=2),inverse_pX_peps:wp.array(dtype=wp.mat33f,ndim=2),IM:wp.array(dtype=wp.mat33f),
                  F:wp.array(dtype=wp.mat33f),E:wp.array(dtype=wp.mat33f)):
    idx = wp.tid()
    whichQuadrature = idx%8
    hex = idx//8
    for i in range(8):
        F[idx] += wp.outer(x[hexagons[hex][i]],shapeFuncGrad[i][whichQuadrature])
    F[idx] = F[idx] @ inverse_pX_peps[hex][whichQuadrature]
    E[idx] = 0.5*(wp.transpose(F[idx])@F[idx]-IM[0])

@wp.kernel()
def compute_elastic_hessian(x:wp.array(dtype=wp.vec3f),hexagons:wp.array(dtype=wp.int32,ndim=2),shapeFuncGradMat:wp.array(dtype=wp.mat33f,ndim=3),shapeFuncGrad:wp.array(dtype=wp.vec3f,ndim=2),
                            F:wp.array(dtype=wp.mat33f),E:wp.array(dtype=wp.mat33f),
                   det_pX_peps:wp.array(dtype=wp.float32,ndim=2),inverse_pX_peps:wp.array(dtype=wp.mat33f,ndim=2),
                   IM:wp.array(dtype=wp.mat33f),LameMu:wp.array(dtype=wp.float32),LameLa:wp.array(dtype=wp.float32),
                   MF_value:wp.array(dtype=wp.mat33f),offset:wp.array(dtype=wp.int32)):
    idx = wp.tid()
    hex = idx//64
    idx_ = idx%64
    whichQuadrature = idx_//8
    whichShapeFunc = idx_%8
    Fid = hex*8+whichQuadrature
    for ii in range(8):
        for i in range(3):
            dF = shapeFuncGradMat[ii][whichQuadrature][i]@inverse_pX_peps[hex][whichQuadrature]
            dE = (wp.transpose(dF)@F[Fid]+wp.transpose(F[Fid])@dF)*0.5
            dP = dF@(2.0*LameMu[0]*E[Fid]+LameLa[0]*wp.trace(E[Fid])*IM[0])+F[Fid]@(2.0*LameMu[0]*dE+LameLa[0]*wp.trace(dE)*IM[0])
            temAns = dP@(wp.transpose(inverse_pX_peps[hex][whichQuadrature]))@wp.transpose(shapeFuncGradMat[whichShapeFunc][whichQuadrature][i])*det_pX_peps[hex][whichQuadrature] 
            wp.atomic_add(MF_value,offset[hex*64+whichShapeFunc*8+ii],temAns) 


@wp.kernel()
def compute_partial_elastic_energy_X(x:wp.array(dtype=wp.vec3),hexagons:wp.array(dtype=wp.int32,ndim=2),vertex2index:wp.array(dtype=wp.int32),
                                     F:wp.array(dtype=wp.mat33f),E:wp.array(dtype=wp.mat33f),
                shapeFuncGrad:wp.array(dtype=wp.vec3f,ndim=2),det_pX_peps:wp.array(dtype=wp.float32,ndim=2),inverse_pX_peps:wp.array(dtype=wp.mat33f,ndim=2),
                IM:wp.array(dtype=wp.mat33f),LameMu:wp.array(dtype=wp.float32),LameLa:wp.array(dtype=wp.float32),
                grad:wp.array(dtype=wp.vec3)):
    idx = wp.tid()
    whichQuadrature = idx%8
    hex = idx//8
    P = F[idx]@(2.0*LameMu[0]*E[idx]+LameLa[0]*wp.trace(E[idx])*IM[0])
    for i in range(8):
        shapeFuncGradNow = shapeFuncGrad[i][whichQuadrature]
        temAns = wp.mul(P@wp.transpose(inverse_pX_peps[hex][whichQuadrature]),shapeFuncGradNow)*det_pX_peps[hex][whichQuadrature]
        wp.atomic_sub(grad,vertex2index[hexagons[hex][i]],temAns)

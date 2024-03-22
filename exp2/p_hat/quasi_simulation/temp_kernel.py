import warp as wp
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
    #F = wp.mat33f()
    F00 = 0.0
    F01 = 0.0
    F02 = 0.0
    F10 = 0.0
    F11 = 0.0
    F12 = 0.0
    F20 = 0.0
    F21 = 0.0
    F22 = 0.0
    for row in range(3):
        for col in range(3):
            value_now = 0.0
            for i in range(8):
                value_now += x[hexagons[hex][i]][row]*shapeFuncGrad[i][whichQuadrature][col]

            F[row,col] = value_now
    for ii in range(8):
        wp.atomic_add(MF_value,offset[hex*64+whichShapeFunc*8+ii],F)
    # F = F @ inverse_pX_peps[hex][whichQuadrature]
    # E = 0.5*(wp.transpose(F)@F-IM[0])

    # shapeFuncGradNow = wp.vec3f()
    # for j in range(3):
    #     shapeFuncGradNow[j] = shapeFuncGrad[whichShapeFunc][whichQuadrature][j]

    # for ii in range(8):
    #     temAnsForHessian = wp.mat33f()
    #     # id0 = hexagons[hex][whichShapeFunc]
    #     # id1 = hexagons[hex][ii]
    #     for i in range(3):
    #         dF = wp.mat33f()
    #         for j in range(3):
    #             dF[i,j] = shapeFuncGrad[ii][whichQuadrature][0]*inverse_pX_peps[hex][whichQuadrature][0,j]+shapeFuncGrad[ii][whichQuadrature][1]*inverse_pX_peps[hex][whichQuadrature][1,j]+shapeFuncGrad[ii][whichQuadrature][2]*inverse_pX_peps[hex][whichQuadrature][2,j]
    #         dE = (wp.transpose(dF)@F+wp.transpose(F)@dF)*0.5
    #         dP = dF@(2.0*LameMu[0]*E+LameLa[0]*wp.trace(E)*IM[0])+F@(2.0*LameMu[0]*dE+LameLa[0]*wp.trace(dE)*IM[0])
    #         temAns = wp.mul(dP@(wp.transpose(inverse_pX_peps[hex][whichQuadrature])),shapeFuncGradNow)*det_pX_peps[hex][whichQuadrature]           
    #         for j in range(3):
    #             temAnsForHessian[j,i] = temAns[j]
    #     wp.atomic_add(MF_value,offset[hex*64+whichShapeFunc*8+ii],temAnsForHessian)
        
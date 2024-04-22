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

@wp.kernel
def prepare_mass(vol:wp.array(dtype=wp.float32),mass:wp.array(dtype=wp.float32),hex:wp.array(dtype=wp.int32,ndim=2)):
    idx = wp.tid()
    whichhex = idx//8
    whichpoint = idx%8
    id = hex[whichhex][whichpoint]
    wp.atomic_add(mass,id,vol[whichhex])

@wp.kernel()
def compute_elastic_energy(x:wp.array(dtype=wp.vec3),hexagons:wp.array(dtype=wp.int32,ndim=2),
                            shapeFuncGrad:wp.array(dtype=wp.float32,ndim=3), det_pX_peps:wp.array(dtype=wp.float32,ndim=2),inverse_pX_peps:wp.array(dtype=wp.mat33f,ndim=2),
                            IM:wp.array(dtype=wp.mat33f),LameMu:wp.array(dtype=wp.float32),LameLa:wp.array(dtype=wp.float32),
                            loss:wp.array(dtype=wp.float32)):
    idx = wp.tid()
    whichQuadrature = idx%8
    hex = idx//8
    F00 = 0.0
    F01 = 0.0
    F02 = 0.0
    F10 = 0.0
    F11 = 0.0
    F12 = 0.0
    F20 = 0.0
    F21 = 0.0
    F22 = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                F00  += x[hexagons[hex][i*4+j*2+k]][0]*shapeFuncGrad[i*4+j*2+k][whichQuadrature][0]      
                F01  += x[hexagons[hex][i*4+j*2+k]][0]*shapeFuncGrad[i*4+j*2+k][whichQuadrature][1]
                F02  += x[hexagons[hex][i*4+j*2+k]][0]*shapeFuncGrad[i*4+j*2+k][whichQuadrature][2]
                F10  += x[hexagons[hex][i*4+j*2+k]][1]*shapeFuncGrad[i*4+j*2+k][whichQuadrature][0]
                F11  += x[hexagons[hex][i*4+j*2+k]][1]*shapeFuncGrad[i*4+j*2+k][whichQuadrature][1]
                F12  += x[hexagons[hex][i*4+j*2+k]][1]*shapeFuncGrad[i*4+j*2+k][whichQuadrature][2]
                F20  += x[hexagons[hex][i*4+j*2+k]][2]*shapeFuncGrad[i*4+j*2+k][whichQuadrature][0]
                F21  += x[hexagons[hex][i*4+j*2+k]][2]*shapeFuncGrad[i*4+j*2+k][whichQuadrature][1]
                F22  += x[hexagons[hex][i*4+j*2+k]][2]*shapeFuncGrad[i*4+j*2+k][whichQuadrature][2]

    FT00 = F00*inverse_pX_peps[hex][whichQuadrature][0][0]+F01*inverse_pX_peps[hex][whichQuadrature][1][0]+F02*inverse_pX_peps[hex][whichQuadrature][2][0]
    FT01 = F00*inverse_pX_peps[hex][whichQuadrature][0][1]+F01*inverse_pX_peps[hex][whichQuadrature][1][1]+F02*inverse_pX_peps[hex][whichQuadrature][2][1]
    FT02 = F00*inverse_pX_peps[hex][whichQuadrature][0][2]+F01*inverse_pX_peps[hex][whichQuadrature][1][2]+F02*inverse_pX_peps[hex][whichQuadrature][2][2]
    FT10 = F10*inverse_pX_peps[hex][whichQuadrature][0][0]+F11*inverse_pX_peps[hex][whichQuadrature][1][0]+F12*inverse_pX_peps[hex][whichQuadrature][2][0]
    FT11 = F10*inverse_pX_peps[hex][whichQuadrature][0][1]+F11*inverse_pX_peps[hex][whichQuadrature][1][1]+F12*inverse_pX_peps[hex][whichQuadrature][2][1]
    FT12 = F10*inverse_pX_peps[hex][whichQuadrature][0][2]+F11*inverse_pX_peps[hex][whichQuadrature][1][2]+F12*inverse_pX_peps[hex][whichQuadrature][2][2]
    FT20 = F20*inverse_pX_peps[hex][whichQuadrature][0][0]+F21*inverse_pX_peps[hex][whichQuadrature][1][0]+F22*inverse_pX_peps[hex][whichQuadrature][2][0]
    FT21 = F20*inverse_pX_peps[hex][whichQuadrature][0][1]+F21*inverse_pX_peps[hex][whichQuadrature][1][1]+F22*inverse_pX_peps[hex][whichQuadrature][2][1]
    FT22 = F20*inverse_pX_peps[hex][whichQuadrature][0][2]+F21*inverse_pX_peps[hex][whichQuadrature][1][2]+F22*inverse_pX_peps[hex][whichQuadrature][2][2]

    E00 = 0.5*(FT00*FT00+FT10*FT10+FT20*FT20-1.0)
    E01 = 0.5*(FT00*FT01+FT10*FT11+FT20*FT21)
    E02 = 0.5*(FT00*FT02+FT10*FT12+FT20*FT22)
    E10 = 0.5*(FT01*FT00+FT11*FT10+FT21*FT20)
    E11 = 0.5*(FT01*FT01+FT11*FT11+FT21*FT21-1.0)
    E12 = 0.5*(FT01*FT02+FT11*FT12+FT21*FT22)
    E20 = 0.5*(FT02*FT00+FT12*FT10+FT22*FT20)
    E21 = 0.5*(FT02*FT01+FT12*FT11+FT22*FT21)
    E22 = 0.5*(FT02*FT02+FT12*FT12+FT22*FT22-1.0)

    Sum = 0.0
    Sum += E00*E00+E01*E01+E02*E02+E10*E10+E11*E11+E12*E12+E20*E20+E21*E21+E22*E22
    trace = E00+E11+E22
    Psi = Sum*LameMu[0]+0.5*LameLa[0]*trace*trace

    energy = Psi*det_pX_peps[hex][whichQuadrature]     

    wp.atomic_add(loss,0,energy)

@wp.kernel()
def compute_gravity_energy(x:wp.array(dtype=wp.vec3),m:wp.array(dtype=wp.float32),g:wp.array(dtype=wp.float32),pin:wp.array(dtype=wp.int32),pin_pos:wp.array(dtype=wp.vec3),control_mag:float,loss:wp.array(dtype=wp.float32)):
    idx = wp.tid()
    energy = -m[idx]*g[0]*x[idx][1]
    if pin[idx] == 1:
        delta_x = pin_pos[idx] - x[idx]
        energy += 0.5*control_mag*wp.dot(delta_x,delta_x)
    wp.atomic_add(loss,0,energy)

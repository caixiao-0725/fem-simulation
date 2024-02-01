import torch
import numpy as np
import pyvista as pv
import warp as wp
import matplotlib.pyplot as plt



def calShapeFuncGrad(shapeFuncGrad,help,quadrature):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                whichShapeFunc = i*4+j*2+k
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            whichQuadrature = 4*a+2*b+c
                            shapeFuncGrad[whichShapeFunc][whichQuadrature][0]=help[whichShapeFunc][0]*(1+help[whichShapeFunc][1]*quadrature[whichQuadrature][1])*(1+help[whichShapeFunc][2]*quadrature[whichQuadrature][2])*0.125
                            shapeFuncGrad[whichShapeFunc][whichQuadrature][1]=help[whichShapeFunc][1]*(1+help[whichShapeFunc][0]*quadrature[whichQuadrature][0])*(1+help[whichShapeFunc][2]*quadrature[whichQuadrature][2])*0.125
                            shapeFuncGrad[whichShapeFunc][whichQuadrature][2]=help[whichShapeFunc][2]*(1+help[whichShapeFunc][1]*quadrature[whichQuadrature][1])*(1+help[whichShapeFunc][0]*quadrature[whichQuadrature][0])*0.125



def prepare(x,N_hexagons,hexagons,shapeFuncGrad,det_pX_peps,inverse_pX_peps):
    with torch.no_grad():
        for hex in range(N_hexagons):
            count_quadrature = 0
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        whichQuadrature = 4*a+2*b+c
                        F = torch.zeros((3,3),dtype=torch.float32)
                        for row in range(3):
                            for col in range(3):
                                value_now = 0.0
                                for i in range(2):
                                    for j in range(2):
                                        for k in range(2):
                                            value_now += x[hexagons[hex][i*4+j*2+k]][row]*shapeFuncGrad[i*4+j*2+k][whichQuadrature][col]
                                F[row][col] = value_now
                        det_pX_peps[hex][count_quadrature] = F.det()
                        inverse_pX_peps[hex][count_quadrature] = F.inverse()
                        count_quadrature += 1

wp.init()
wp.set_device('cuda:0')


@wp.kernel()
def compute_elastic_energy(x:wp.array(dtype=wp.vec3),F:wp.array(dtype=wp.float32,ndim=3),A:wp.array(dtype=wp.float32,ndim=3),E:wp.array(dtype=wp.float32,ndim=3),
                   hexagons:wp.array(dtype=wp.int32,ndim=2),shapeFuncGrad:wp.array(dtype=wp.float32,ndim=3),
                   det_pX_peps:wp.array(dtype=wp.float32,ndim=2),inverse_pX_peps:wp.array(dtype=wp.mat33f,ndim=2),
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
def compute_gravity_energy(x:wp.array(dtype=wp.vec3),m:wp.array(dtype=wp.float32),g:wp.array(dtype=wp.float32),loss:wp.array(dtype=wp.float32)):
    idx = wp.tid()
    energy = -m[0]*g[0]*x[idx][1]
    wp.atomic_add(loss,0,energy)

@wp.kernel()
def pin(x:wp.array(dtype=wp.vec3),pin_pos:wp.array(dtype=wp.vec3),pin_list:wp.array(dtype=wp.int32)):
    idx = wp.tid()
    x[pin_list[idx]] = pin_pos[idx]




def qSim(mesh_path,dx,pinList):

    # init once and for all
    # voxelize
    mesh = pv.read(mesh_path)
    voxels = pv.voxelize(mesh, density=dx, check_surface=False)
    hex = []
    N = voxels.points.shape[0]
    for cell in voxels.cell:
        hex.append([cell.point_ids[0],cell.point_ids[4],cell.point_ids[3],cell.point_ids[7],cell.point_ids[1],cell.point_ids[5],cell.point_ids[2],cell.point_ids[6]])
    N_hexagons = len(hex)

    # simulation components
    x = torch.tensor(voxels.points,dtype=torch.float32,requires_grad=True)

    # pinned components
    pin_pos = torch.zeros((len(pinList),3),dtype=torch.float32,requires_grad=False)
    pin_list = torch.zeros((len(pinList)),dtype=torch.int32,requires_grad=False)
    for i in range(len(pinList)):
        pin_list[i] = pinList[i]
        pin_pos[i]=x[pinList[i]].clone().detach()

    # geometric components
    hexagons = torch.zeros((N_hexagons,8),dtype=torch.int32,requires_grad=False)
    for i in range(N_hexagons):
        for j in range(8):
            hexagons[i][j] = hex[i][j]

    # prepare
    shapeFuncGrad = torch.zeros((8,8,3),dtype=torch.float32)
    help = torch.tensor([[-1,-1,-1],
                        [-1,-1,1],
                        [-1,1,-1],
                        [-1,1,1],
                        [1,-1,-1],
                        [1,-1,1],
                        [1,1,-1],
                        [1,1,1]],dtype=torch.float32)



    quadrature = torch.tensor([[-0.57735,-0.57735,-0.57735],
                            [-0.57735,-0.57735,0.57735],
                            [-0.57735,0.57735,-0.57735],
                            [-0.57735,0.57735,0.57735],
                            [0.57735,-0.57735,-0.57735],
                            [0.57735,-0.57735,0.57735],
                            [0.57735,0.57735,-0.57735],
                            [0.57735,0.57735,0.57735]])

    calShapeFuncGrad(shapeFuncGrad,help,quadrature)

    #for general hex,we need patial X/patial epsilon on the 8 quadrature points . Often we use the inverse of the matrix,so we save the inverse.
    inverse_pX_peps = torch.zeros((N_hexagons,8,3,3),dtype=torch.float32)
    det_pX_peps = torch.zeros((N_hexagons,8),dtype=torch.float32)
    IM = torch.eye(3,dtype=torch.float32)

    prepare(x,N_hexagons,hexagons,shapeFuncGrad,det_pX_peps,inverse_pX_peps)

    x_cpu = wp.from_torch(x,dtype=wp.vec3)
    x_gpu = wp.zeros_like(x_cpu,device='cuda:0',requires_grad=True)
    wp.copy(x_gpu,x_cpu)

    inverse_pX_peps_torch_gpu = inverse_pX_peps.to('cuda:0')
    det_pX_peps_torch_gpu = det_pX_peps.to('cuda:0')
    IM_torch_gpu = IM.to('cuda:0')
    hexagons_torch_gpu = hexagons.to('cuda:0')
    shapeFuncGrad_torch_gpu = shapeFuncGrad.to('cuda:0')
    pin_pos = pin_pos.to('cuda:0')
    pin_list = pin_list.to('cuda:0')

    inverse_pX_peps_gpu = wp.from_torch(inverse_pX_peps_torch_gpu,dtype=wp.mat33f)
    det_pX_peps_gpu = wp.from_torch(det_pX_peps_torch_gpu)
    IM_gpu = wp.from_torch(IM_torch_gpu,dtype=wp.mat33f)
    hexagons_gpu = wp.from_torch(hexagons_torch_gpu)
    shapeFuncGrad_gpu = wp.from_torch(shapeFuncGrad_torch_gpu)
    pin_pos_gpu = wp.from_torch(pin_pos,dtype=wp.vec3)
    pin_list_gpu = wp.from_torch(pin_list)

    # physical quantities
    LameMu_gpu = wp.array([3e6],dtype=wp.float32,requires_grad=False,device='cuda:0')
    LameLa_gpu = wp.array([0.0],dtype=wp.float32,requires_grad=False,device='cuda:0')
    m_gpu = wp.array([1.0],dtype=wp.float32,requires_grad=False,device='cuda:0')
    g_gpu = wp.array([-9.8],dtype=wp.float32,requires_grad=False,device='cuda:0')
    F_gpu = wp.array(shape=(N_hexagons*8,3,3),dtype=wp.float32,requires_grad=False,device='cuda:0')
    cacheA_gpu = wp.array(shape=(N_hexagons*8,3,3),dtype=wp.float32,requires_grad=False,device='cuda:0')
    cacheB_gpu = wp.array(shape=(N_hexagons*8,3,3),dtype=wp.float32,requires_grad=False,device='cuda:0')

    loss = wp.zeros((1),dtype=wp.float32,requires_grad=True,device='cuda:0')
    tape = wp.Tape()
    with tape:
        wp.launch(kernel=compute_gravity_energy,dim=N,inputs=[x_gpu,m_gpu,g_gpu,loss])
        wp.launch(kernel=compute_elastic_energy,dim=N_hexagons*8,inputs=[x_gpu,F_gpu,cacheA_gpu,cacheB_gpu,hexagons_gpu,shapeFuncGrad_gpu,det_pX_peps_gpu,inverse_pX_peps_gpu,IM_gpu,LameMu_gpu,LameLa_gpu,loss])
        
    optimizer = torch.optim.Adam([wp.to_torch(x_gpu)], lr=1e-3)
    plot_x = []
    plot_y = []
    tape.zero()
    for step in range(64000):
        tape.zero()
        tape.backward(loss)
        optimizer.step()
        wp.launch(kernel=pin,dim=len(pinList),inputs=[x_gpu,pin_pos_gpu,pin_list_gpu])
        loss.zero_()
        wp.launch(kernel=compute_gravity_energy,dim=N,inputs=[x_gpu,m_gpu,g_gpu,loss])
        wp.launch(kernel=compute_elastic_energy,dim=N_hexagons*8,inputs=[x_gpu,F_gpu,cacheA_gpu,cacheB_gpu,hexagons_gpu,shapeFuncGrad_gpu,det_pX_peps_gpu,inverse_pX_peps_gpu,IM_gpu,LameMu_gpu,LameLa_gpu,loss])

        if step % 1000 == 0 and step != 0:
            print('step {}: f(x)={}'.format(step,loss.numpy()[0]))
            plot_x.append(step)
            plot_y.append(loss.numpy()[0])

    # 创建画布和子图
    fig, ax = plt.subplots()

    # 绘制线图
    ax.plot(plot_x, plot_y, linestyle='-', color='blue', label='Line')

    # 添加标题和标签
    ax.set_title('Plot')
    ax.set_xlabel('iterations')
    ax.set_ylabel('energy')

    # 添加图例
    ax.legend()

    # 显示图形
    plt.show()

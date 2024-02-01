import torch
import numpy as np
import pyvista as pv
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
                            shapeFuncGrad[i][j][k][a][b][c][0]=help[whichShapeFunc][0]*(1+help[whichShapeFunc][1]*quadrature[whichQuadrature][1])*(1+help[whichShapeFunc][2]*quadrature[whichQuadrature][2])*0.125
                            shapeFuncGrad[i][j][k][a][b][c][1]=help[whichShapeFunc][1]*(1+help[whichShapeFunc][0]*quadrature[whichQuadrature][0])*(1+help[whichShapeFunc][2]*quadrature[whichQuadrature][2])*0.125
                            shapeFuncGrad[i][j][k][a][b][c][2]=help[whichShapeFunc][2]*(1+help[whichShapeFunc][1]*quadrature[whichQuadrature][1])*(1+help[whichShapeFunc][0]*quadrature[whichQuadrature][0])*0.125



def prepare(x,N_hexagons,hexagons,shapeFuncGrad,det_pX_peps,inverse_pX_peps):
    with torch.no_grad():
        for hex in range(N_hexagons):
            count_quadrature = 0
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        F = torch.zeros((3,3),dtype=torch.float32)
                        for row in range(3):
                            for col in range(3):
                                value_now = 0.0
                                for i in range(2):
                                    for j in range(2):
                                        for k in range(2):
                                            value_now += x[hexagons[hex][i*4+j*2+k]][row]*shapeFuncGrad[i][j][k][a][b][c][col]
                                F[row][col] = value_now
                        det_pX_peps[hex][count_quadrature] = F.det()
                        inverse_pX_peps[hex][count_quadrature] = F.inverse()
                        count_quadrature += 1


def compute_energy(x,N,N_hexagons,hexagons,shapeFuncGrad,det_pX_peps,inverse_pX_peps,IM,LameMu,LameLa,m,g):
    first = True
    for hex in range(N_hexagons):
        count_quadrature = 0

        for a in range(2):
            for b in range(2):
                for c in range(2):
                    F = torch.zeros((3,3),dtype=torch.float32)
                    for row in range(3):
                        for col in range(3):
                            value_now = 0.0
                            for i in range(2):
                                for j in range(2):
                                    for k in range(2):
                                        value_now += x[hexagons[hex][i*4+j*2+k]][row]*shapeFuncGrad[i][j][k][a][b][c][col]
                            F[row][col] = value_now
                    F = F @ inverse_pX_peps[hex][count_quadrature]
                    E = 0.5*(F.transpose(0,1)@F-IM)
                    Psi = torch.sum(torch.square(E))*LameMu + 0.5*LameLa*E.trace()*E.trace()
                    if first:
                        energy = Psi*det_pX_peps[hex,count_quadrature]
                        first = False
                    else:
                        energy = energy + Psi*det_pX_peps[hex,count_quadrature]     
                    count_quadrature+=1
    for i in range(N):
        energy = energy -m*g*x[i][1] 
    return energy
    


# init once and for all
    
def qSim(mesh_path,dx,pinList):
    # physical quantities
    m = 1.0
    g = -9.8

    LameMu = torch.tensor(3e6,dtype=torch.float32)
    LameLa = torch.tensor(0.0,dtype=torch.float32)



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
    pin_pos = []
    for i in range(len(pinList)):
        pin_pos.append(x[pinList[i]].clone().detach())
    # geometric components
    hexagons = torch.zeros((N_hexagons,8),dtype=torch.int32,requires_grad=False)
    for i in range(N_hexagons):
        for j in range(8):
            hexagons[i][j] = hex[i][j]

    # prepare
    shapeFuncGrad = torch.zeros((2,2,2,2,2,2,3),dtype=torch.float32)
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
    avg_det_pX_peps = torch.zeros((N_hexagons),dtype=torch.float32)
    IM = torch.eye(3,dtype=torch.float32)

    prepare(x,N_hexagons,hexagons,shapeFuncGrad,det_pX_peps,inverse_pX_peps)

    optimizer = torch.optim.Adam([x], lr=1e-3)
    plot_x = []
    plot_y = []
    for step in range(1000):
        energy = compute_energy(x,N,N_hexagons,hexagons,shapeFuncGrad,det_pX_peps,inverse_pX_peps,IM,LameMu,LameLa,m,g)
        optimizer.zero_grad()
        energy.backward()
        #print(x.grad)
        optimizer.step()
        with torch.no_grad():
            for i in range(len(pinList)):
                x[pinList[i]] = pin_pos[i]
        if step % 50 == 0:
            print('step {}: f(x)={}'.format(step,energy.item()))
            plot_x.append(step)
            plot_y.append(energy.item())

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

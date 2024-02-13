import pyvista as pv
import warp as wp
import warp.sparse
import warp.optim.linear
import torch            #warp只能在gpu上运行，所以这里用torch来做一些cpu上的操作再运到warp上
import numpy as np
from hexagon import *
from cublas import *
import time
import sys

class Object:
    MF_L_row_gpu : wp.array(dtype=wp.int32)
    MF_L_col_gpu : wp.array(dtype=wp.int32)
    MF_D_row_gpu : wp.array(dtype=wp.int32)
    MF_D_col_gpu : wp.array(dtype=wp.int32)
    MF_U_row_gpu : wp.array(dtype=wp.int32)
    MF_U_col_gpu : wp.array(dtype=wp.int32)
    MF_row_gpu : wp.array(dtype=wp.int32)
    MF_col_gpu : wp.array(dtype=wp.int32)
    diag_offset_gpu : wp.array(dtype=wp.int32)

    MF_value_gpu : wp.array(dtype=wp.mat33f)
    MF_L_value_gpu : wp.array(dtype=wp.mat33f)
    MF_D_value_gpu : wp.array(dtype=wp.mat33f)
    MF_U_value_gpu : wp.array(dtype=wp.mat33f)

    hex_update_offset_gpu : wp.array(dtype=wp.int32)

    def __init__(self,mesh_path,dx,pinList):
        self.mesh = pv.read(mesh_path)
        self.dx = dx
        self.voxels = pv.voxelize(self.mesh, density=dx, check_surface=False)
        hex = []
        self.N_verts = self.voxels.points.shape[0]
        for cell in self.voxels.cell:
            hex.append([cell.point_ids[0],cell.point_ids[4],cell.point_ids[3],cell.point_ids[7],cell.point_ids[1],cell.point_ids[5],cell.point_ids[2],cell.point_ids[6]])
        self.N_hexagons = len(hex)
        print('Num of hexagons : ',self.N_hexagons)
        # simulation components
        x = torch.tensor(self.voxels.points,dtype=torch.float32,requires_grad=True)

        # max_idx = -1
        # max_value = -1000.0
        # for i in range(x.shape[0]):
        #     if x[i][1]>max_value:
        #         max_value = x[i][1]
        #         max_idx = i 
        # print(max_idx)

        # pinned components
        pin_pos = torch.zeros((len(pinList),3),dtype=torch.float32,requires_grad=False)
        pin_list = torch.zeros((len(pinList)),dtype=torch.int32,requires_grad=False)
        self.N_pin = len(pinList)
        for i in range(len(pinList)):
            pin_list[i] = pinList[i]
            pin_pos[i]=x[pinList[i]].clone().detach()

        # geometric components
        my_dict = {}

        hexagons = torch.zeros((self.N_hexagons,8),dtype=torch.int32,requires_grad=False)

        for i in range(self.N_hexagons):
            for j in range(8):
                hexagons[i][j] = hex[i][j]
        hex_update_offset = torch.zeros((self.N_hexagons*64),dtype=torch.int32,requires_grad=False)
        diag_offset = torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False)
        
        self.store_LDU = 1
        self.MF_nnz = 0
        self.MF_D_nnz = 0
        self.MF_L_nnz = 0
        self.MF_U_nnz = 0
        self.off_l = 0 
        self.off_d = 0
        self.off_u = 0


        
        if self.store_LDU == 1:
            for i in range(self.N_hexagons):
                for j in range(8):
                    for k in range(8):
                        id0 = hex[i][j]
                        id1 = hex[i][k]
                        key = tuple([id0,id1])
                        if key not in my_dict:
                            my_dict[key] = 0
            self.MF_nnz = len(my_dict)
            self.MF_D_nnz = self.N_verts
            self.MF_L_nnz = (len(my_dict)-self.N_verts)>>1
            self.MF_U_nnz = self.MF_L_nnz
            self.off_l = 0 
            self.off_d = self.MF_L_nnz
            self.off_u = self.MF_L_nnz+self.MF_D_nnz
            self.MF_L_row = torch.zeros((self.MF_L_nnz),dtype=torch.int32,requires_grad=False)
            self.MF_L_col = torch.zeros((self.MF_L_nnz),dtype=torch.int32,requires_grad=False)
            self.MF_D_row = torch.zeros((self.MF_D_nnz),dtype=torch.int32,requires_grad=False)
            self.MF_D_col = torch.zeros((self.MF_D_nnz),dtype=torch.int32,requires_grad=False)
            self.MF_U_row = torch.zeros((self.MF_U_nnz),dtype=torch.int32,requires_grad=False)
            self.MF_U_col = torch.zeros((self.MF_U_nnz),dtype=torch.int32,requires_grad=False)
            i_l = 0
            i_d = 0
            i_u = 0
            new_dict = {}
            for i in range(self.N_hexagons):
                for j in range(8):
                    for k in range(8):
                        id0 = hex[i][j]
                        id1 = hex[i][k]
                        key = tuple([id0,id1])
                        if id0>id1:
                            self.MF_L_row[i_l] = id0
                            self.MF_L_col[i_l] = id1
                            if key not in new_dict:
                                new_dict[key] = i_l + self.off_l
                                hex_update_offset[i*64+j*8+k] = i_l + self.off_l
                                i_l += 1
                            else:
                                hex_update_offset[i*64+j*8+k]=new_dict[key]
                        if id0 == id1:
                            self.MF_D_row[id0] = id0
                            self.MF_D_col[id1] = id1
                            if key not in new_dict:
                                new_dict[key] = id0 + self.off_d
                                hex_update_offset[i*64+j*8+k] = id0 + self.off_d
                                i_d += 1
                            else:
                                hex_update_offset[i*64+j*8+k]=new_dict[key]
                        if id0<id1:
                            self.MF_U_row[i_u] = id0
                            self.MF_U_col[i_u] = id1
                            if key not in new_dict:
                                new_dict[key] = i_u + self.off_u
                                hex_update_offset[i*64+j*8+k] = i_u + self.off_u
                                i_u += 1
                            else:
                                hex_update_offset[i*64+j*8+k]=new_dict[key]

            self.MF_L_value_gpu = wp.zeros(shape=(self.MF_L_nnz),dtype=wp.mat33f,device='cuda:0')
            self.MF_D_value_gpu = wp.zeros(shape=(self.MF_D_nnz),dtype=wp.mat33f,device='cuda:0')
            self.MF_U_value_gpu = wp.zeros(shape=(self.MF_U_nnz),dtype=wp.mat33f,device='cuda:0')
            self.MF_D_row_gpu = wp.from_torch(self.MF_D_row.to('cuda:0'),dtype=wp.int32)
            self.MF_D_col_gpu = wp.from_torch(self.MF_D_col.to('cuda:0'),dtype=wp.int32)
            self.MF_L_row_gpu = wp.from_torch(self.MF_L_row.to('cuda:0'),dtype=wp.int32)
            self.MF_L_col_gpu = wp.from_torch(self.MF_L_col.to('cuda:0'),dtype=wp.int32)
            self.MF_U_row_gpu = wp.from_torch(self.MF_U_row.to('cuda:0'),dtype=wp.int32)
            self.MF_U_col_gpu = wp.from_torch(self.MF_U_col.to('cuda:0'),dtype=wp.int32)
            self.hex_update_offset_gpu = wp.from_torch(hex_update_offset.to('cuda:0'),dtype=wp.int32)

        else:
            i_f = 0
            for i in range(self.N_hexagons):
                for j in range(8):
                    for k in range(8):
                        id0 = hex[i][j]
                        id1 = hex[i][k]
                        key = tuple([id0,id1])
                        if key not in my_dict:
                            my_dict[key] = i_f
                            hex_update_offset[i*64+j*8+k] = i_f
                            i_f += 1
                        else:
                            hex_update_offset[i*64+j*8+k]=my_dict[key]
                        if id0 == id1:
                            diag_offset[id0] = my_dict[key]

            self.MF_nnz = i_f
            
            self.MF_row_gpu = wp.zeros(shape=(self.MF_nnz),dtype=wp.int32,device='cuda:0')
            self.MF_col_gpu = wp.zeros(shape=(self.MF_nnz),dtype=wp.int32,device='cuda:0')
            row = np.zeros((self.MF_nnz),dtype=np.int32)
            col = np.zeros((self.MF_nnz),dtype=np.int32)
            for i in my_dict:
                row[my_dict[i]] = i[0]
                col[my_dict[i]] = i[1]
            MF_row_cpu = wp.from_numpy(row)
            MF_col_cpu = wp.from_numpy(col)
            wp.copy(self.MF_row_gpu,MF_row_cpu)
            wp.copy(self.MF_col_gpu,MF_col_cpu)

            self.diag_offset_gpu = wp.from_torch(diag_offset.to('cuda:0'),dtype=wp.int32)
            self.hex_update_offset_gpu = wp.from_torch(hex_update_offset.to('cuda:0'),dtype=wp.int32)

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
        IM = torch.eye(3,dtype=torch.float32)

        self.x_cpu = wp.from_torch(x,dtype=wp.vec3)
        self.x_gpu = wp.zeros_like(self.x_cpu,device='cuda:0')
        self.x_lineSearch = wp.zeros_like(self.x_cpu,device='cuda:0')
        self.grad_gpu = wp.zeros_like(self.x_cpu,device='cuda:0')
        wp.copy(self.x_gpu,self.x_cpu)
        wp.copy(self.x_lineSearch,self.x_cpu)

        IM_torch_gpu = IM.to('cuda:0')
        hexagons_torch_gpu = hexagons.to('cuda:0')
        shapeFuncGrad_torch_gpu = shapeFuncGrad.to('cuda:0')
        pin_pos = pin_pos.to('cuda:0')
        pin_list = pin_list.to('cuda:0')


        self.IM_gpu = wp.from_torch(IM_torch_gpu,dtype=wp.mat33f)
        self.hexagons_gpu = wp.from_torch(hexagons_torch_gpu)
        self.shapeFuncGrad_gpu = wp.from_torch(shapeFuncGrad_torch_gpu)
        self.pin_pos_gpu = wp.from_torch(pin_pos,dtype=wp.vec3)
        self.pin_list_gpu = wp.from_torch(pin_list)
        


        # physical quantities
        self.LameMu_gpu = wp.array([3e5],dtype=wp.float32,requires_grad=False,device='cuda:0')
        self.LameLa_gpu = wp.array([0.0],dtype=wp.float32,requires_grad=False,device='cuda:0')
        self.m_gpu = wp.array([1.0],dtype=wp.float32,requires_grad=False,device='cuda:0')
        self.g_gpu = wp.array([-9.8],dtype=wp.float32,requires_grad=False,device='cuda:0')

        self.energy = wp.zeros((1),dtype=wp.float32,requires_grad=False,device='cuda:0')
        self.energy_lineSearch = wp.zeros((1),dtype=wp.float32,requires_grad=False,device='cuda:0')

        self.inverse_pX_peps_gpu = wp.array(shape = (self.N_hexagons,8),dtype=wp.mat33f)
        self.det_pX_peps_gpu = wp.array(shape = (self.N_hexagons,8),dtype=wp.float32)

        wp.launch(kernel=prepare_kernal,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu])

        #build constrains
        
        
        #wp.launch(kernel=build_constraints,dim=self.N_hexagons,inputs=[self.hexagons_gpu,self.hex_update_offset_gpu])
        self.A = warp.sparse.bsr_zeros(self.N_verts,self.N_verts,wp.mat33f,device='cuda:0')
        self.L = warp.sparse.bsr_zeros(self.N_verts,self.N_verts,wp.mat33f,device='cuda:0')
        self.U = warp.sparse.bsr_zeros(self.N_verts,self.N_verts,wp.mat33f,device='cuda:0')
        self.x = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')

        self.MF_value_gpu = wp.zeros(shape=(self.MF_nnz),dtype=wp.mat33f,device='cuda:0')
        self.dev_temp_X = wp.zeros(shape=(self.N_verts),dtype=wp.vec3,device='cuda:0')


    def step(self):
        #wp.launch(kernel=compute_gravity_energy,dim=N,inputs=[x_gpu,m_gpu,g_gpu,loss])
        start = time.time()
        self.grad_gpu.fill_(0.0)
        wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy])
        wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_gpu,self.m_gpu,self.g_gpu,self.energy])

        wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.grad_gpu])
        wp.launch(kernel=compute_partial_gravity_energy_X,dim=self.N_verts,inputs=[self.m_gpu,self.g_gpu,self.grad_gpu])
        end = time.time()
        print('Time : ',end-start)
        print('Energy : ',self.energy.numpy()[0])

    def compute_A_and_B(self):
        start = time.time()
        wp.launch(kernel=compute_elastic_hessian,dim=self.N_hexagons*64,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.MF_value_gpu,self.hex_update_offset_gpu])
        wp.synchronize()
        end = time.time()
        print('Time : ',end-start)


    def gradientDescent(self,iterations = 100,lr = 1e-3):
        for iter in range(iterations):
            self.grad_gpu.zero_()
            self.energy.zero_()
            wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy])
            wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_gpu,self.m_gpu,self.g_gpu,self.energy])
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.grad_gpu])
            wp.launch(kernel=compute_partial_gravity_energy_X,dim=self.N_verts,inputs=[self.m_gpu,self.g_gpu,self.grad_gpu])
            alpha = lr
            for i in range(20):
                wp.launch(kernel=minues_grad,dim=self.N_verts,inputs=[self.x_gpu,self.x_lineSearch,self.grad_gpu,alpha])
                wp.launch(kernel=pin,dim=self.N_pin,inputs=[self.x_lineSearch,self.pin_pos_gpu,self.pin_list_gpu])
                self.energy_lineSearch.zero_()
                wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_lineSearch,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy_lineSearch])
                wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_lineSearch,self.m_gpu,self.g_gpu,self.energy_lineSearch])
                wp.synchronize()
                if self.energy_lineSearch.numpy()[0] < self.energy.numpy()[0]:
                    wp.copy(self.x_gpu,self.x_lineSearch)
                    break
                else:
                    alpha = alpha*0.2
            wp.synchronize()
            if iter%(iterations/100) == 0:
                print('iter :',iter,' Energy : ',self.energy.numpy()[0])

    def Newton(self,iterations = 50):
        for step in range(iterations):
            #if step%(iterations/10) == 0:
            if 1:
                wp.synchronize()
                print('Step : ',step)
                self.energy.zero_()
                wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy])
                wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_gpu,self.m_gpu,self.g_gpu,self.energy])
                print('Energy : ',self.energy.numpy()[0])
            self.grad_gpu.zero_()
            self.MF_value_gpu.zero_()
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.grad_gpu])
            wp.launch(kernel=compute_partial_gravity_energy_X,dim=self.N_verts,inputs=[self.m_gpu,self.g_gpu,self.grad_gpu])
            wp.launch(kernel=compute_elastic_hessian,dim=self.N_hexagons*64,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.MF_value_gpu,self.hex_update_offset_gpu])
            wp.launch(kernel=spd_matrix33f,dim=self.MF_nnz,inputs=[self.MF_value_gpu])
            #self.x.zero_()
            if self.store_LDU == 0:
                wp.launch(kernel=jacobi_iteration_offset,dim=self.N_verts,inputs=[self.x,self.MF_value_gpu,self.diag_offset_gpu,self.grad_gpu])
            else:
                wp.copy(self.MF_L_value_gpu,self.MF_value_gpu,dest_offset=0,src_offset=self.off_l,count=self.MF_L_nnz)
                wp.copy(self.MF_D_value_gpu,self.MF_value_gpu,dest_offset=0,src_offset=self.off_d,count=self.MF_D_nnz)
                wp.copy(self.MF_U_value_gpu,self.MF_value_gpu,dest_offset=0,src_offset=self.off_u,count=self.MF_U_nnz)
                warp.sparse.bsr_set_from_triplets(self.L,self.MF_L_row_gpu,self.MF_L_col_gpu,self.MF_L_value_gpu)
                warp.sparse.bsr_set_from_triplets(self.U,self.MF_U_row_gpu,self.MF_U_col_gpu,self.MF_U_value_gpu)
                for jacobiIter in range(1):
                    wp.copy(self.dev_temp_X,self.grad_gpu)
                    warp.sparse.bsr_mv(self.L,self.x,self.dev_temp_X,alpha=-1.0,beta=1.0)
                    warp.sparse.bsr_mv(self.U,self.x,self.dev_temp_X,alpha=-1.0,beta=1.0)
                    wp.launch(kernel=jacobi_iteration,dim=self.N_verts,inputs=[self.x,self.MF_D_value_gpu,self.dev_temp_X])
                    
            #print(self.x)
            wp.launch(kernel=subVec3,dim=self.N_verts,inputs=[self.x_gpu,self.x])
            wp.launch(kernel=pin,dim=self.N_pin,inputs=[self.x_gpu,self.pin_pos_gpu,self.pin_list_gpu])

    def show(self):
        wp.copy(self.x_cpu,self.x_gpu)
        plot_verts = self.x_cpu.numpy()
        self.voxels.points = plot_verts
        pl = pv.Plotter()  
        pl.add_mesh(self.voxels, color=True, show_edges=True)
        # Set camera start position
        pl.camera_position = 'xy'
        pl.show()  
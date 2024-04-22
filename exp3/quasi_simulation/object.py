import pyvista as pv
import warp as wp
from sparse import *
import torch            #warp只能在gpu上运行，所以这里用torch来做一些cpu上的操作再运到warp上
import numpy as np
from quasi_hexagon_3 import *
from cublas import *
from cpu_function import *
import time
import sys
import operator
import matplotlib.pyplot as plt
from model import *

from cuda import cudart
from OpenGL.GL import *

EPSILON = 1e-7

class torch_dynamic(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,delta_x,num_vert,N_hexagons,m,g,pin_gpu,all_pin_pos_gpu,control_mag,hexagons_gpu,shapeFuncGrad_gpu,det_pX_peps_gpu,inverse_pX_peps_gpu,IM_gpu,LameMu_gpu,LameLa_gpu):
        ctx.x = wp.from_torch(x,dtype=wp.vec3f)
        ctx.delta_x = wp.from_torch(delta_x,dtype=wp.vec3f)
        ctx.num_vert = num_vert
        ctx.N_hexagons = N_hexagons
        ctx.m = m
        ctx.g = g
        ctx.pin_gpu = pin_gpu
        ctx.all_pin_pos_gpu = all_pin_pos_gpu
        ctx.control_mag = control_mag
        ctx.hexagons_gpu = hexagons_gpu
        ctx.shapeFuncGrad_gpu = shapeFuncGrad_gpu
        ctx.det_pX_peps_gpu = det_pX_peps_gpu
        ctx.inverse_pX_peps_gpu = inverse_pX_peps_gpu
        ctx.IM_gpu = IM_gpu
        ctx.LameMu_gpu = LameMu_gpu
        ctx.LameLa_gpu = LameLa_gpu
        ctx.loss = wp.zeros((1),dtype=wp.float32,requires_grad=True,device='cuda:0')
        ctx.tape = wp.Tape()
        
        with ctx.tape:
            wp.launch(kernel=axpy,dim=ctx.num_vert,inputs=[ctx.delta_x,ctx.x,1.0])
            wp.launch(kernel=compute_gravity_energy,dim=ctx.num_vert,inputs=[ctx.delta_x,ctx.m,ctx.g,ctx.pin_gpu,ctx.all_pin_pos_gpu,ctx.control_mag,ctx.loss])
            wp.launch(kernel=compute_elastic_energy,dim=ctx.N_hexagons*8,inputs=[ctx.delta_x,ctx.hexagons_gpu,ctx.shapeFuncGrad_gpu,ctx.det_pX_peps_gpu,ctx.inverse_pX_peps_gpu,ctx.IM_gpu,ctx.LameMu_gpu,ctx.LameLa_gpu,ctx.loss])

        return (wp.to_torch(ctx.delta_x),wp.to_torch(ctx.loss))
    
    @staticmethod
    def backward(ctx,adj_delta_x,adj_loss):
        ctx.delta_x.grad = wp.from_torch(adj_delta_x,dtype=wp.vec3f)
        ctx.loss.grad = wp.from_torch(adj_loss,dtype=wp.float32)
        ctx.tape.backward()

        return (None,wp.to_torch(ctx.tape.gradients[ctx.delta_x]),None,None,None,None,None,None,None,None,None,None,None,None,None,None)

class Object: 

    def __init__(self,mesh_path,dx,pinList):
        # 层数
        self.layer = 1
        self.control_mag = 100.0
        self.spd_value = 1e-3

        #读取mesh并且体素化    存储hex索引和顶点位置
        self.mesh = pv.read(mesh_path)
        self.dx = dx
        self.voxels = pv.voxelize(self.mesh, density=dx, check_surface=False)
        hex = []
        self.N_verts = self.voxels.points.shape[0]
        for cell in self.voxels.cell:
            hex.append([cell.point_ids[0],cell.point_ids[4],cell.point_ids[3],cell.point_ids[7],cell.point_ids[1],cell.point_ids[5],cell.point_ids[2],cell.point_ids[6]])
        self.N_hexagons = len(hex)
        
        print('Num of hexagons : ',self.N_hexagons)
        
        edge_hash = {}
        for i in range(self.N_hexagons):
            for j in range(8):
                for k in range(j+1,8):
                    id0 = hex[i][j]
                    id1 = hex[i][k]
                    if id0>id1:
                        id0,id1 = id1,id0
                    if (id0,id1) not in edge_hash:
                        edge_hash[(id0,id1)]=1

        self.edge_num = len(edge_hash)
        self.edge_index = torch.zeros((2,self.edge_num),dtype=torch.long)
        for i,(id0,id1) in enumerate(edge_hash.keys()):
            self.edge_index[0][i] = id0
            self.edge_index[1][i] = id1
        self.edge_index = self.edge_index.to('cuda:0')
        # simulation components
        x = torch.tensor(self.voxels.points,dtype=torch.float32,requires_grad=False)
        hexagons = torch.zeros((self.N_hexagons,8),dtype=torch.int32,requires_grad=False)
        self.x_torch = x.clone().detach().to('cuda:0')

        for i in range(self.N_hexagons):
            for j in range(8):
                hexagons[i][j] = hex[i][j]
                #算出表面索引,用于opengl渲染
        hex_face = [[0,2,3,1],[4,5,7,6],[0,1,5,4],[2,6,7,3],[0,4,6,2],[1,3,7,5]]
        hex_order_face = [[0,1,3,2],[4,5,7,6],[0,4,5,1],[2,6,7,3],[0,4,6,2],[1,5,7,3]]
        face_dict = {}
        count = 0
        for h in hex:
            for i in range(6):
                face = [h[hex_face[i][0]],h[hex_face[i][1]],h[hex_face[i][2]],h[hex_face[i][3]]]
                key = tuple(face)
                if key not in face_dict:
                    face_dict[key] = 6*count+i
                else :
                    face_dict[key] = -1
            count += 1
        
        self.surface_face = []
        for key in face_dict.keys():
            if face_dict[key] >= 0:
                i = face_dict[key]%6
                h = hex[face_dict[key]//6]
                f0 = h[hex_order_face[i][0]]
                f1 = h[hex_order_face[i][1]]
                f2 = h[hex_order_face[i][2]]
                f3 = h[hex_order_face[i][3]]
                self.surface_face.append(f0)
                self.surface_face.append(f1)
                self.surface_face.append(f2)
                self.surface_face.append(f0)
                self.surface_face.append(f2)
                self.surface_face.append(f3)
        
        self.surface_face = np.array(self.surface_face,dtype=np.int32)
        self.N_face = int(self.surface_face.size/3)
        self.face_gpu = wp.from_numpy(self.surface_face,dtype = wp.vec3i,device='cuda:0')
        self.face_normal_gpu = wp.zeros((self.N_face),dtype = wp.vec3f,device='cuda:0')
        #申请opengl相关的资源
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        self.EBO = glGenBuffers(1)
        self.NBO = glGenBuffers(1)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, 4*self.N_verts*3, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        glBindBuffer(GL_ARRAY_BUFFER, self.NBO)
        glBufferData(GL_ARRAY_BUFFER, 4*self.N_verts*3, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * self.surface_face.size,self.surface_face , GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)

        vbo_graphics_ressource = check_cudart_err(cudart.cudaGraphicsGLRegisterBuffer( self.VBO, cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))
        check_cudart_err(cudart.cudaGraphicsMapResources(1, vbo_graphics_ressource,None))
        vert_ptr, size = check_cudart_err(cudart.cudaGraphicsResourceGetMappedPointer(vbo_graphics_ressource))
        check_cudart_err(cudart.cudaGraphicsUnmapResources(1, vbo_graphics_ressource, None))

        self.render_x = wp.array(ptr=vert_ptr,length=self.N_verts,shape =None,dtype=wp.vec3f,device='cuda:0')


        nbo_graphics_ressource = check_cudart_err(cudart.cudaGraphicsGLRegisterBuffer( self.NBO, cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))
        check_cudart_err(cudart.cudaGraphicsMapResources(1, nbo_graphics_ressource,None))
        norm_ptr, size = check_cudart_err(cudart.cudaGraphicsResourceGetMappedPointer(nbo_graphics_ressource))
        check_cudart_err(cudart.cudaGraphicsUnmapResources(1, nbo_graphics_ressource, None))

        self.vert_normal_gpu = wp.array(ptr=norm_ptr,length=self.N_verts,shape =None,dtype=wp.vec3f,device='cuda:0')


        vertex2index = []
        index2vertex = []
        vertex2index.append(torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False))
        index2vertex.append(torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False))
        
        #高斯赛德尔 color 
        #空间六面体的color数量应该固定是8个
        self.color_num = []
        self.color_vertex_num = []
        self.color_num.append(8)
        self.color_vertex_num.append([0 for _ in range(self.color_num[0]+1)])
        color = torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False)

        #找出包围盒的顶点，用于计算相对位置
        self.box_min = torch.tensor([1000.0,1000.0,1000.0],dtype=torch.float32)
        self.box_max = torch.tensor([-1000.0,-1000.0,-1000.0],dtype=torch.float32)
        for i in range(self.N_verts):
            for j in range(3):
                if x[i][j]<self.box_min[j]:
                    self.box_min[j] = x[i][j]
                if x[i][j]>self.box_max[j]:
                    self.box_max[j] = x[i][j]
        
        #算出包围盒三个维度的最小值，用来确定多重网格法层数的大小
        delta_box = self.box_max-self.box_min
        min_box = torch.min(delta_box)
        min_box = torch.log(min_box/self.dx)/torch.log(torch.tensor(2.0))
        self.layer = min_box.int()-1
        if self.layer < 1:
            self.layer = 1
        
        
        
        for i in range(self.N_verts):
            ind = color_ind(x[i],self.box_min,self.dx)
            color[i] = ind
        
        index_idx = 0
        for j in range(self.color_num[0]):
            for i in range(self.N_verts):    
                if color[i] == j:
                    vertex2index[0][i] = index_idx
                    index2vertex[0][index_idx] = i
                    index_idx += 1
            self.color_vertex_num[0][j+1] = index_idx

        # fine to coarse
        self.dims = []
        self.hexs = []
        self.x = []
        self.fine_coarse = []
        self.dims.append(self.N_verts)
        self.hexs.append(hexagons)
        self.x.append(x)


            
        for layer in range(self.layer-1):
            coarse_hex_num = 0
            coarse_vert_num = 0
            #粗网格的网格大小是细网格的两倍
            if layer == 0:
                fine_size = self.dx
            else:
                fine_size = self.dx*2
            coarse_size = fine_size*2

            #建立hash表 对于每一个立方体，算出它的中心对应的整数索引进行分类
            hex_hash = {}
        
            for i in range(self.hexs[layer].shape[0]):
                
                #计算中心 hex里面第一个点是最小的点，所以直接加上dx/2就是中心
                ijk = ijk_index(self.x[layer][self.hexs[layer][i][0]],self.box_min,coarse_size)
                hex_key = tuple([ijk[0],ijk[1],ijk[2]])

                if hex_key not in hex_hash:
                    hex_hash[hex_key] = coarse_hex_num
                    coarse_hex_num += 1

            coarse_hex = torch.zeros((coarse_hex_num,8),dtype=torch.int32,requires_grad=False)
            vert_hash = {}
            for hex_key in hex_hash.keys():
                hex_id = hex_hash[hex_key]
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            vert_key = tuple([hex_key[0]+j,hex_key[1]+k,hex_key[2]+l])
                            if vert_key not in vert_hash:
                                vert_hash[vert_key] = coarse_vert_num
                                coarse_vert_num += 1
                            vert_id = vert_hash[vert_key]
                            coarse_hex[hex_id][4*j+2*k+l] = vert_id
            self.dims.append(coarse_vert_num)
            self.hexs.append(coarse_hex)
            fine_coarse  = torch.zeros((coarse_hex_num,27),dtype=torch.int32,requires_grad=False)
            fine_coarse.fill_(-1)
            for i in range(self.hexs[layer].shape[0]):
                for j in range(8):
                    ijk_fine = ijk_index(self.x[layer][self.hexs[layer][i][j]],self.box_min,fine_size)
                    ijk_coarse = ijk_index(self.x[layer][self.hexs[layer][i][0]],self.box_min,coarse_size) 
                    hex_key = tuple([ijk_coarse[0],ijk_coarse[1],ijk_coarse[2]])
                    hex_id = hex_hash[hex_key]
                    relative = [ijk_fine[0]-2*ijk_coarse[0],ijk_fine[1]-2*ijk_coarse[1],ijk_fine[2]-2*ijk_coarse[2]]
                    fine_coarse[hex_id][9*relative[0]+3*relative[1]+relative[2]] = self.hexs[layer][i][j]
            self.fine_coarse.append(fine_coarse)
            # color GS
            self.color_num.append(8)
            self.color_vertex_num.append([0 for _ in range(self.color_num[layer+1]+1)])
            vertex2index.append(torch.zeros((coarse_vert_num),dtype=torch.int32,requires_grad=False))
            index2vertex.append(torch.zeros((coarse_vert_num),dtype=torch.int32,requires_grad=False))

            color = torch.zeros((coarse_vert_num),dtype=torch.int32,requires_grad=False)
            coarse_vert = torch.zeros((coarse_vert_num,3),dtype=torch.float32,requires_grad=False)
            for vert_key in vert_hash.keys():
                vert_id = vert_hash[vert_key]
                coarse_vert[vert_id] = torch.tensor(vert_key,dtype=torch.float32)*coarse_size+self.box_min
                ind = 4*(vert_key[2]%2)+2*(vert_key[1]%2)+(vert_key[0]%2)
                color[vert_id] = ind
            self.x.append(coarse_vert)
            index_idx = 0
            for j in range(self.color_num[layer+1]):
                for i in range(coarse_vert_num):    
                    if color[i] == j:
                        vertex2index[layer+1][i] = index_idx
                        index2vertex[layer+1][index_idx] = i
                        index_idx += 1
                self.color_vertex_num[layer+1][j+1] = index_idx
            
        print('coarse hex build done')


        #compute U Matrix  
        #有hat的是用于计算残差的，没有hat的是用于计算预条件矩阵的  没有order的是不用GS的,有order的是有GS的
        self.Ut = [bsr_zeros(self.dims[i+1],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer-1)]
        self.Us = [bsr_zeros(self.dims[i],self.dims[i+1],wp.mat33f,device='cuda:0') for i in range(self.layer-1)]

        self.Ut_noOrder = [bsr_zeros(self.dims[i+1],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer-1)]
        self.Us_noOrder = [bsr_zeros(self.dims[i],self.dims[i+1],wp.mat33f,device='cuda:0') for i in range(self.layer-1)]

        self.Ut_hat = [bsr_zeros(self.dims[i+1],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer-1)]
        self.Us_hat = [bsr_zeros(self.dims[i],self.dims[i+1],wp.mat33f,device='cuda:0') for i in range(self.layer-1)]

        self.Ut_noOrder_hat = [bsr_zeros(self.dims[i+1],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer-1)]
        self.Us_noOrder_hat = [bsr_zeros(self.dims[i],self.dims[i+1],wp.mat33f,device='cuda:0') for i in range(self.layer-1)]
        
        fix_idx = torch.zeros((self.N_verts,8),dtype=torch.int32,requires_grad=False)
        fix_idx.fill_(-1)
        fix_num = torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False)
        fix_values = torch.zeros((self.N_verts,8),dtype=torch.float32,requires_grad=False)
        
                                      
        self.fix_idx_gpu = wp.from_torch(fix_idx.to('cuda:0'),dtype=wp.int32)
        self.fix_values_gpu = wp.from_torch(fix_values.to('cuda:0'),dtype=wp.float32)

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
        self.x_gpu_layer = [wp.zeros(self.dims[i],dtype=wp.vec3,requires_grad=False,device='cuda:0') for i in range(self.layer)]
        wp.copy(self.x_gpu_layer[0],self.x_cpu)
        for i in range(1,self.layer):
            bsr_mv(self.Ut_noOrder[i-1],self.x_gpu_layer[i-1],self.x_gpu_layer[i],alpha=1.0,beta=0.0)
        self.x_lineSearch = wp.zeros_like(self.x_cpu,device='cuda:0')
        self.grad_gpu = wp.zeros_like(self.x_cpu,device='cuda:0')
        wp.copy(self.x_lineSearch,self.x_cpu)

        self.IM_gpu = wp.from_torch(IM.to('cuda:0'),dtype=wp.mat33f)
        self.hexagons_gpu = [wp.from_torch(self.hexs[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer)]#wp.from_torch(hexagons.to('cuda:0'))
        self.shapeFuncGrad_gpu = wp.from_torch(shapeFuncGrad.to('cuda:0'))
        self.dev_vertex2index = [wp.from_torch(vertex2index[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer)]        
        self.dev_index2vertex = [wp.from_torch(index2vertex[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer)]

        # physical quantities
        self.LameMu_gpu = wp.array([250.0],dtype=wp.float32,requires_grad=False,device='cuda:0')
        self.LameLa_gpu = wp.array([0],dtype=wp.float32,requires_grad=False,device='cuda:0')
        self.m_gpu = [wp.zeros((self.dims[i]),dtype=wp.float32,requires_grad=False,device='cuda:0') for i in range(self.layer)]
        self.vol = [wp.zeros((self.hexs[i].shape[0]),dtype=wp.float32) for i in range(self.layer)]
        #wp.launch(kernel=prepare_mass,dim = self.N_hexagons*8,inputs=[self.vol[0],self.m_gpu[0],self.hexagons_gpu[0]])
        self.g_gpu = wp.array([-9.8/2.0],dtype=wp.float32,requires_grad=False,device='cuda:0')

        self.energy = wp.zeros((1),dtype=wp.float32,requires_grad=False,device='cuda:0')
        self.energy_lineSearch = wp.zeros((1),dtype=wp.float32,requires_grad=False,device='cuda:0')

        self.inverse_pX_peps_gpu = [wp.array(shape = (self.hexs[i].shape[0],8),dtype=wp.mat33f) for i in range(self.layer)]
        self.det_pX_peps_gpu = [wp.array(shape = (self.hexs[i].shape[0],8),dtype=wp.float32) for i in range(self.layer)]

        for i in range(1):
            wp.launch(kernel=prepare_kernal,dim=self.hexs[i].shape[0]*8,inputs=[self.x_gpu_layer[i],self.hexagons_gpu[i],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[i],self.inverse_pX_peps_gpu[i]])
            wp.launch(kernel=Sum_8,dim = self.hexs[i].shape[0],inputs = [self.vol[i],self.det_pX_peps_gpu[i]])
            wp.launch(kernel=prepare_mass,dim=self.hexs[i].shape[0]*8,inputs=[self.vol[i],self.m_gpu[i],self.hexagons_gpu[i]])
        # pinned components
        if len(pinList) == 0:
            max_idx = -1
            max_value = -1000.0
            for i in range(x.shape[0]):
                if x[i][1]>max_value:
                    max_value = x[i][1]
                    max_idx = i 
            #print(max_idx)
            
            for i in range(x.shape[0]):
                if x[i][1]>= max_value-self.dx-1e-5:
                    pinList.append(i)
        else:
            x[pinList[0]][1] += 0.0
            wp.copy(self.x_gpu_layer[0],self.x_cpu)

        pin_cpu = torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False)
        for i in range(len(pinList)):
            pin_cpu[pinList[i]] = 1

        pin_pos = torch.zeros((len(pinList),3),dtype=torch.float32,requires_grad=False)
        pin_list = torch.zeros((len(pinList)),dtype=torch.int32,requires_grad=False)
        self.pin_gpu = wp.from_torch(pin_cpu.to('cuda:0'),dtype=wp.int32)
        self.all_pin_pos_gpu = wp.from_torch(x.to('cuda:0'),dtype=wp.vec3f)

        self.N_pin = len(pinList)
        for i in range(len(pinList)):
            pin_list[i] = pinList[i]
            pin_pos[i]=x[pinList[i]].clone().detach()         
        self.pin_pos_gpu = wp.from_torch(pin_pos.to('cuda:0'),dtype=wp.vec3)
        self.pin_list_gpu = wp.from_torch(pin_list.to('cuda:0'))

        # 矩阵 A = L + U + D
        self.A = [bsr_zeros(self.dims[i],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer)]
        self.L = [bsr_zeros(self.dims[i],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer)]
        self.U = [bsr_zeros(self.dims[i],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer)]
        self.D = [bsr_zeros(self.dims[i],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer)]

        self.MF_GS_U = [bsr_zeros(self.color_vertex_num[0][i+1]-self.color_vertex_num[0][i],self.color_vertex_num[0][self.color_num[0]]-self.color_vertex_num[0][i+1],wp.mat33f,device='cuda:0') for i in range(self.color_num[0]-1)]
        self.MF_GS_L = [bsr_zeros(self.color_vertex_num[0][i+2]-self.color_vertex_num[0][i+1],self.color_vertex_num[0][i+1],wp.mat33f,device='cuda:0') for i in range(self.color_num[0]-1)]
        
        self.X = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.dev_E = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_R = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_P = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_AP = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_B = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_B_fixed = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_delta_x = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_x_solved = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        #adam的系数
        self.m = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.v = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.m_hat = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.v_hat = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')


        self.dev_temp_X = wp.zeros(shape=(self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.squared_sum = wp.zeros(shape=(1),dtype=wp.float32,device='cuda:0')
        self.dot_sum = wp.zeros(shape=(1),dtype=wp.float32,device='cuda:0')
        self.norm_max = wp.zeros(shape=(1),dtype=wp.float32,device='cuda:0')

    def Adam(self,iterations = 1000,lr = 1e-3,beta1 = 0.9,beta2 = 0.999,epsilon = 1e-8):
        '''
        Adam optimization algorithm
        只用一阶项求解非线性方程组
        '''
        self.x_gpu_layer[0].requires_grad = True
        loss = wp.zeros((1),dtype=wp.float32,requires_grad=True,device='cuda:0')
        tape = wp.Tape()
        with tape:
            wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.m_gpu[0],self.g_gpu,self.pin_gpu,self.all_pin_pos_gpu,self.control_mag,loss])
            wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,loss])
            
        optimizer = torch.optim.Adam([wp.to_torch(self.x_gpu_layer[0])], lr=1e-3)
        for step in range(1,iterations+1):
            tape.zero()
            tape.backward(loss)
            optimizer.step()
            loss.zero_()
            wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.m_gpu[0],self.g_gpu,self.pin_gpu,self.all_pin_pos_gpu,self.control_mag,loss])
            wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,loss])
            if step % 100 == 0 and step != 0:
                print('step {}: f(x)={}'.format(step,loss.numpy()[0]))

    def train(self,iterations = 1000,lr = 1e-3,beta1 = 0.9,beta2 = 0.999,epsilon = 1e-8):
        '''
        简单的test
        '''
        #loss = wp.zeros((1),dtype=wp.float32,requires_grad=True,device='cuda:0')
        temp_x = wp.to_torch(self.x_gpu_layer[0])
        input_x = torch.cat((temp_x,self.x_torch),dim=1)

        net = MDN3().to('cuda:0')
        for param in net.parameters():
            param.requires_grad = True
        delta_x = net(input_x,self.edge_index)

        te_x,loss = torch_dynamic.apply(temp_x,delta_x,self.N_verts,self.N_hexagons,self.m_gpu[0],self.g_gpu,self.pin_gpu,self.all_pin_pos_gpu,self.control_mag,self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu)

        print('f(x)={}'.format(loss.item()))

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        for step in range(1,iterations+1):
            #print(delta_x)
            loss.backward()
            optimizer.step()
            loss.zero_()
            optimizer.zero_grad()

            wp.copy(self.x_gpu_layer[0],self.x_cpu)
            delta_x = net(input_x,self.edge_index)

            te_x,loss = torch_dynamic.apply(temp_x,delta_x,self.N_verts,self.N_hexagons,self.m_gpu[0],self.g_gpu,self.pin_gpu,self.all_pin_pos_gpu,self.control_mag,self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu)
            #if step % 100 == 0 and step != 0:
            print('step {}: f(x)={}'.format(step,loss.item()))
        wp.copy(self.x_gpu_layer[0],wp.from_torch(te_x))

    def updateNormal(self):
        self.vert_normal_gpu.zero_()
        wp.launch(kernel = updateFaceNorm,dim = self.N_face,inputs = [self.x_gpu_layer[0],self.face_gpu,self.face_normal_gpu])
        wp.launch(kernel= updateVertNorm,dim = self.N_face,inputs = [self.vert_normal_gpu,self.face_gpu,self.face_normal_gpu])

    def render(self,pause = False):
        if not pause:
            self.dev_B_fixed[0].zero_()
            self.dev_B_fixed[1].zero_()

            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.dev_delta_x[0],self.dev_index2vertex[0]])
        wp.copy(self.render_x,self.x_gpu_layer[0])
        self.updateNormal()
        wp.synchronize()
        glDrawElements(GL_TRIANGLES, len(self.surface_face), GL_UNSIGNED_INT, None)


    def show_layer(self,layer=0):
        index = np.array([0,4,6,2,1,5,7,3])
        cells = self.hexs[layer].numpy()[:,index]
        hex_num = np.zeros((cells.shape[0],1),dtype=np.int32)
        hex_num.fill(8)
        cells = np.concatenate((hex_num,cells),axis=1)
        cell_type = np.zeros(cells.shape[0],dtype=np.int8)
        cell_type.fill(pv.CellType.HEXAHEDRON)
        wp.copy(self.x_cpu,self.x_gpu_layer[layer])
        grid = pv.UnstructuredGrid(cells, cell_type, self.x[layer].numpy())
        grid.plot(show_edges=True)

    def show(self):
        # 创建画布和子图    
        fig, (ax1,ax2) = plt.subplots(2)
        fig.set_figheight(9)
        fig.set_figwidth(13)

        # 绘制线图
        ax1.plot(self.plot_x, self.plot_energy, linestyle='-', color='blue', label='Line')

        # 添加标题和标签
        ax1.set_title('Energy')
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('energy')

        # 绘制线图
        ax2.plot(self.plot_x, self.plot_InfNorm, linestyle='-', color='blue', label='Line')

        # 添加标题和标签
        ax2.set_title('InfNorm')
        ax2.set_xlabel('iterations')
        ax2.set_ylabel('InfNorm')

        # 添加图例
        # ax1.legend()

        # 显示图形
        plt.show()

        wp.copy(self.x_cpu,self.x_gpu_layer[0])
        plot_verts = self.x[0].numpy()
        self.voxels.points = plot_verts
        pl = pv.Plotter()  
        pl.add_mesh(self.voxels, color=True, show_edges=True)
        # Set camera start position
        pl.camera_position = 'xy'
        pl.show()  
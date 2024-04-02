import pyvista as pv
import warp as wp
from sparse import *
import torch            #warp只能在gpu上运行，所以这里用torch来做一些cpu上的操作再运到warp上
import numpy as np
from train_hexagon import *
from temp_kernel import *
from cublas import *
from cpu_function import *
import time
import sys
import operator
import matplotlib.pyplot as plt


from cuda import cudart
from OpenGL.GL import *

EPSILON = 1e-7
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
        print('Num of vertices : ',self.N_verts)
        # simulation components
        x = torch.tensor(self.voxels.points,dtype=torch.float32,requires_grad=False)
        hexagons = torch.zeros((self.N_hexagons,8),dtype=torch.int32,requires_grad=False)
        
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
        
        self.surface_vert = []
        surface_hash = {}
        for v in self.surface_face:
            if v not in surface_hash:
                surface_hash[v] = 0
                self.surface_vert.append(v)
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
        

        for layer in range(self.layer-1):
            rowInd = []
            colInd = []
            rowInd_noOrder = []
            colInd_noOrder = []
            value = []
            fine_hash = {}
            for HEX in range(self.fine_coarse[layer].shape[0]):
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            fine = 9*i+3*j+k
                            fine_id = self.fine_coarse[layer][HEX][fine].item()
                            if fine_id != -1:
                                if fine_id not in fine_hash:
                                    fine_hash[fine_id] = 0

                                    #8个顶点的插值，插值系数是1，最好插值
                                    if i%2+j%2 +k%2 == 0:
                                        coarse_id = self.hexs[layer+1][HEX][4*(i//2)+2*(j//2)+(k//2)]
                                        rowInd.append(vertex2index[layer+1][coarse_id])
                                        colInd.append(vertex2index[layer][fine_id])
                                        rowInd_noOrder.append(coarse_id)
                                        colInd_noOrder.append(fine_id)
                                        if layer == 0:
                                            fix_idx[fine_id][fix_num[fine_id]] = coarse_id
                                            fix_values[fine_id][fix_num[fine_id]] = 1.0
                                            fix_num[fine_id] += 1
                                        value.append(1.0)
                                    #最中间那个点，插值系数是0.125，给8个顶点插值
                                    elif i%2+j%2 +k%2 == 3:
                                        for l in range(8):
                                            coarse_id = self.hexs[layer+1][HEX][l]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            rowInd_noOrder.append(coarse_id)
                                            colInd_noOrder.append(fine_id)
                                            if layer == 0:
                                                fix_idx[fine_id][fix_num[fine_id]] = coarse_id
                                                fix_values[fine_id][fix_num[fine_id]] = 0.125
                                                fix_num[fine_id] += 1
                                            value.append(0.125)
                                    # 12个边的中点，插值系数是0.5，给2个顶点插值
                                    elif i%2+j%2+k%2 == 1:
                                        if i == 1:
                                            coarse_id = self.hexs[layer+1][HEX][4*0+2*(j//2)+(k//2)]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            rowInd_noOrder.append(coarse_id)
                                            colInd_noOrder.append(fine_id)
                                            if layer == 0:
                                                fix_idx[fine_id][fix_num[fine_id]] = coarse_id
                                                fix_values[fine_id][fix_num[fine_id]] = 0.5
                                                fix_num[fine_id] += 1
                                            value.append(0.5)
                                            coarse_id = self.hexs[layer+1][HEX][4*1+2*(j//2)+(k//2)]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            rowInd_noOrder.append(coarse_id)
                                            colInd_noOrder.append(fine_id)
                                            if layer == 0:
                                                fix_idx[fine_id][fix_num[fine_id]] = coarse_id
                                                fix_values[fine_id][fix_num[fine_id]] = 0.5
                                                fix_num[fine_id] += 1
                                            value.append(0.5)
                                        elif j == 1:
                                            coarse_id = self.hexs[layer+1][HEX][4*(i//2)+2*0+(k//2)]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            rowInd_noOrder.append(coarse_id)
                                            colInd_noOrder.append(fine_id)
                                            if layer == 0:
                                                fix_idx[fine_id][fix_num[fine_id]] = coarse_id
                                                fix_values[fine_id][fix_num[fine_id]] = 0.5
                                                fix_num[fine_id] += 1
                                            value.append(0.5)
                                            coarse_id = self.hexs[layer+1][HEX][4*(i//2)+2*1+(k//2)]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            rowInd_noOrder.append(coarse_id)
                                            colInd_noOrder.append(fine_id)
                                            if layer == 0:
                                                fix_idx[fine_id][fix_num[fine_id]] = coarse_id
                                                fix_values[fine_id][fix_num[fine_id]] = 0.5
                                                fix_num[fine_id] += 1
                                            value.append(0.5)
                                        elif k == 1:
                                            coarse_id = self.hexs[layer+1][HEX][4*(i//2)+2*(j//2)+0]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            rowInd_noOrder.append(coarse_id)
                                            colInd_noOrder.append(fine_id)
                                            if layer == 0:
                                                fix_idx[fine_id][fix_num[fine_id]] = coarse_id
                                                fix_values[fine_id][fix_num[fine_id]] = 0.5
                                                fix_num[fine_id] += 1
                                            value.append(0.5)
                                            coarse_id = self.hexs[layer+1][HEX][4*(i//2)+2*(j//2)+1]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            rowInd_noOrder.append(coarse_id)
                                            colInd_noOrder.append(fine_id)
                                            if layer == 0:
                                                fix_idx[fine_id][fix_num[fine_id]] = coarse_id
                                                fix_values[fine_id][fix_num[fine_id]] = 0.5
                                                fix_num[fine_id] += 1
                                            value.append(0.5)
                                    #6个面的中心，给4个顶点插值，插值系数是0.25       
                                    elif i%2+j%2+k%2 == 2:
                                        if i%2==0:
                                            for ii in range(2):
                                                for jj in range(2):
                                                    coarse_id = self.hexs[layer+1][HEX][4*(i//2)+2*ii+jj]
                                                    rowInd.append(vertex2index[layer+1][coarse_id])
                                                    colInd.append(vertex2index[layer][fine_id])
                                                    rowInd_noOrder.append(coarse_id)
                                                    colInd_noOrder.append(fine_id)
                                                    if layer == 0:
                                                        fix_idx[fine_id][fix_num[fine_id]] = coarse_id
                                                        fix_values[fine_id][fix_num[fine_id]] = 0.25
                                                        fix_num[fine_id] += 1
                                                    value.append(0.25)
                                        elif j%2==0:
                                            for ii in range(2):
                                                for jj in range(2):
                                                    coarse_id = self.hexs[layer+1][HEX][4*ii+2*(j//2)+jj]
                                                    rowInd.append(vertex2index[layer+1][coarse_id])
                                                    colInd.append(vertex2index[layer][fine_id])
                                                    rowInd_noOrder.append(coarse_id)
                                                    colInd_noOrder.append(fine_id)
                                                    if layer == 0:
                                                        fix_idx[fine_id][fix_num[fine_id]] = coarse_id
                                                        fix_values[fine_id][fix_num[fine_id]] = 0.25
                                                        fix_num[fine_id] += 1
                                                    value.append(0.25)
                                        elif k%2==0:
                                            for ii in range(2):
                                                for jj in range(2):
                                                    coarse_id = self.hexs[layer+1][HEX][4*ii+2*jj+(k//2)]
                                                    rowInd.append(vertex2index[layer+1][coarse_id])
                                                    colInd.append(vertex2index[layer][fine_id])
                                                    rowInd_noOrder.append(coarse_id)
                                                    colInd_noOrder.append(fine_id)
                                                    if layer == 0:
                                                        fix_idx[fine_id][fix_num[fine_id]] = coarse_id
                                                        fix_values[fine_id][fix_num[fine_id]] = 0.25
                                                        fix_num[fine_id] += 1
                                                    value.append(0.25)
            Us_rowInd = wp.from_torch(torch.tensor(rowInd,dtype=torch.int32).to('cuda:0'))
            Us_colInd = wp.from_torch(torch.tensor(colInd,dtype=torch.int32).to('cuda:0'))
            Us_rowInd_noOrder = wp.from_torch(torch.tensor(rowInd_noOrder,dtype=torch.int32).to('cuda:0'))
            Us_colInd_noOrder = wp.from_torch(torch.tensor(colInd_noOrder,dtype=torch.int32).to('cuda:0'))
            norm = torch.zeros((self.dims[layer+1]),dtype=torch.float32)
            for i in range(len(value)):
                row = rowInd_noOrder[i]           
                norm[row] += value[i]

            Us_value = torch.zeros((len(value),3,3),dtype=torch.float32)
            for i in range(len(value)):
                row = rowInd_noOrder[i]
                Us_value[i] = torch.eye(3,dtype=torch.float32)*value[i]/norm[row]
            
            Us_hat_value = torch.zeros((len(value),3,3),dtype=torch.float32)
            for i in range(len(value)):
                Us_hat_value[i] = torch.eye(3,dtype=torch.float32)*value[i]
            
            Us_values = wp.from_torch(Us_value.to('cuda:0'),dtype=wp.mat33f) 
            Us_hat_values = wp.from_torch(Us_hat_value.to('cuda:0'),dtype=wp.mat33f) 
            bsr_set_from_triplets(self.Ut[layer],Us_rowInd,Us_colInd,Us_values)
            bsr_set_from_triplets_with_grad(self.Ut_noOrder[layer],Us_rowInd_noOrder,Us_colInd_noOrder,Us_values)
            bsr_set_from_triplets(self.Ut_hat[layer],Us_rowInd,Us_colInd,Us_hat_values)
            bsr_set_from_triplets(self.Ut_noOrder_hat[layer],Us_rowInd_noOrder,Us_colInd_noOrder,Us_hat_values)
        
        self.Ut_noOrder_value = wp.zeros((self.Ut_noOrder[0].nnz),dtype=wp.float32,requires_grad=True)
        wp.launch(kernel = p_hat_init,dim =self.Ut_noOrder[0].nnz ,inputs=[self.Ut_noOrder_value,self.Ut_noOrder[0].values])
        

        #Us和Ut这样写是为了和tiantian liu的论文对齐
        for layer in range(self.layer-1):
            self.Us[layer] = bsr_transposed(self.Ut[layer])
            self.Us_noOrder[layer] = bsr_transposed(self.Ut_noOrder[layer])
            self.Us_hat[layer] = bsr_transposed(self.Ut_hat[layer])
            self.Us_noOrder_hat[layer] = bsr_transposed(self.Ut_noOrder_hat[layer])
        print('U matrix build done')    
                                      
        self.fix_idx_gpu = wp.from_torch(fix_idx.to('cuda:0'),dtype=wp.int32)
        self.fix_values_gpu = wp.from_torch(fix_values.to('cuda:0'),dtype=wp.float32)

        hex_update_offset = [torch.zeros((self.hexs[i].shape[0]*64),dtype=torch.int32,requires_grad=False) for i in range(self.layer)]
        diag_offset = torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False)
        
        self.store_LDU = 1
        self.MF_nnz = 0
        self.MF_D_nnz = 0
        self.MF_L_nnz = 0
        self.MF_U_nnz = 0
        self.off_l = 0  
        self.off_d = 0
        self.off_u = 0

        self.UtAUs_block_offset = []
        self.UtAUs_nnz = [0 for _ in range(self.layer-1)]
        self.UtAUs_U_nnz = [0 for _ in range(self.layer-1)]
        self.UtAUs_L_nnz = [0 for _ in range(self.layer-1)]
        self.UtAUs_D_nnz = [0 for _ in range(self.layer-1)]
        self.UtAUs_U_row = []
        self.UtAUs_U_col = []
        self.UtAUs_L_row = []
        self.UtAUs_L_col = []
        self.UtAUs_D_row = []
        self.UtAUs_D_col = []
        self.UtAUs_GS_U_col = []
        self.UtAUs_GS_U_row = []
        self.UtAUs_GS_L_col = []
        self.UtAUs_GS_L_row = []
        self.UtAUs_GS_U_Ptr = []
        self.UtAUs_GS_L_Ptr = []
        self.UtAUs_off_l = [0 for _ in range(self.layer-1)]
        self.UtAUs_off_d = [0 for _ in range(self.layer-1)]
        self.UtAUs_off_u = [0 for _ in range(self.layer-1)]
        



        for layer in range(self.layer):
            if layer == 0:
                # geometric components
                my_dict = {}
                for i in range(self.hexs[layer].shape[0]):
                    for j in range(8):
                        for k in range(8):
                            id0 = vertex2index[layer][self.hexs[layer][i][j]].item()
                            id1 = vertex2index[layer][self.hexs[layer][i][k]].item()
                            key = tuple([id0,id1])
                            if key not in my_dict:
                                my_dict[key] = 0
                sorted_dict = dict(sorted(my_dict.items(), key=operator.itemgetter(0)))

                self.MF_nnz = len(my_dict)
                self.MF_D_nnz = self.dims[layer]
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
                MF_GS_L_col = torch.zeros((self.MF_L_nnz),dtype=torch.int32,requires_grad=False)
                MF_GS_L_row = torch.zeros((self.MF_L_nnz),dtype=torch.int32,requires_grad=False)
                MF_GS_U_col = torch.zeros((self.MF_U_nnz),dtype=torch.int32,requires_grad=False)
                MF_GS_U_row = torch.zeros((self.MF_U_nnz),dtype=torch.int32,requires_grad=False)
                self.MF_GS_U_Ptr = [0 for _ in range(self.color_num[0]+1)]
                self.MF_GS_L_Ptr = [0 for _ in range(self.color_num[0]+1)]
                
                i_l = 0
                i_d = 0
                i_u = 0
                p_base_l = -1
                p_base_u = -1
                now_base_l = 0
                next_base_l = self.color_vertex_num[0][1]
                r_base_u = -1
                c_base_u = 0

                for key in sorted_dict.keys():
                    r = key[0]
                    c = key[1]
                    if c == r:
                        self.MF_D_row[i_d] = r
                        self.MF_D_col[i_d] = c
                        sorted_dict[key] = r+self.off_d
                        diag_offset[i_d] = r+self.off_d
                        i_d += 1
                    if r<c:
                        while(r>=c_base_u):
                            p_base_u+=1
                            self.MF_GS_U_Ptr[p_base_u] = i_u
                            r_base_u = self.color_vertex_num[0][p_base_u]
                            c_base_u = self.color_vertex_num[0][p_base_u+1]
                        self.MF_U_row[i_u] = r
                        self.MF_U_col[i_u] = c
                        MF_GS_U_row[i_u] = r-r_base_u
                        MF_GS_U_col[i_u] = c-c_base_u 
                        sorted_dict[key] = i_u+self.off_u
                        i_u += 1
                    if r > c:
                        while(r>=next_base_l):
                            p_base_l+=1
                            self.MF_GS_L_Ptr[p_base_l] = i_l
                            now_base_l = self.color_vertex_num[0][p_base_l+1]
                            next_base_l = self.color_vertex_num[0][p_base_l+2]
                        self.MF_L_row[i_l] = r
                        self.MF_L_col[i_l] = c
                        MF_GS_L_row[i_l] = r-now_base_l
                        MF_GS_L_col[i_l] = c
                        sorted_dict[key] = i_l+self.off_l
                        i_l += 1

                for i in range(self.N_hexagons):
                    for j in range(8):
                        for k in range(8):
                            id0 = vertex2index[0][hex[i][j]].item()
                            id1 = vertex2index[0][hex[i][k]].item()
                            key = tuple([id0,id1])
                            hex_update_offset[0][i*64+j*8+k] = sorted_dict[key]

                while(p_base_l<self.color_num[0]-1):
                    p_base_l+=1
                    self.MF_GS_L_Ptr[p_base_l] = i_l
                while(p_base_u<self.color_num[0]-1):
                    p_base_u+=1
                    self.MF_GS_U_Ptr[p_base_u] = i_u
                # for i in range(self.color_num+1):
                #     print( self.color_vertex_num[i],self.MF_GS_L_Ptr[i],self.MF_GS_U_Ptr[i])
                self.MF_D_row_gpu = wp.from_torch(self.MF_D_row.to('cuda:0'),dtype=wp.int32)
                self.MF_D_col_gpu = wp.from_torch(self.MF_D_col.to('cuda:0'),dtype=wp.int32)
                self.MF_L_row_gpu = wp.from_torch(self.MF_L_row.to('cuda:0'),dtype=wp.int32)
                self.MF_L_col_gpu = wp.from_torch(self.MF_L_col.to('cuda:0'),dtype=wp.int32)
                self.MF_U_row_gpu = wp.from_torch(self.MF_U_row.to('cuda:0'),dtype=wp.int32)
                self.MF_U_col_gpu = wp.from_torch(self.MF_U_col.to('cuda:0'),dtype=wp.int32)
                self.MF_GS_U_col_gpu = wp.from_torch(MF_GS_U_col.to('cuda:0'),dtype=wp.int32)
                self.MF_GS_U_row_gpu = wp.from_torch(MF_GS_U_row.to('cuda:0'),dtype=wp.int32)
                self.MF_GS_L_col_gpu = wp.from_torch(MF_GS_L_col.to('cuda:0'),dtype=wp.int32)
                self.MF_GS_L_row_gpu = wp.from_torch(MF_GS_L_row.to('cuda:0'),dtype=wp.int32)

                self.diag_offset_gpu = wp.from_torch(diag_offset.to('cuda:0'),dtype=wp.int32)                

                self.MF_row_gpu = wp.zeros(shape=(self.MF_nnz),dtype=wp.int32,device='cuda:0')
                self.MF_col_gpu = wp.zeros(shape=(self.MF_nnz),dtype=wp.int32,device='cuda:0')
                wp.copy(self.MF_row_gpu,self.MF_D_row_gpu,self.off_d,0,self.MF_D_nnz)
                wp.copy(self.MF_col_gpu,self.MF_D_col_gpu,self.off_d,0,self.MF_D_nnz)
                wp.copy(self.MF_row_gpu,self.MF_L_row_gpu,self.off_l,0,self.MF_L_nnz)
                wp.copy(self.MF_col_gpu,self.MF_L_col_gpu,self.off_l,0,self.MF_L_nnz)
                wp.copy(self.MF_row_gpu,self.MF_U_row_gpu,self.off_u,0,self.MF_U_nnz)
                wp.copy(self.MF_col_gpu,self.MF_U_col_gpu,self.off_u,0,self.MF_U_nnz)


            else:
                my_dict = {}
                for i in range(self.hexs[layer].shape[0]):
                    for j in range(8):
                        for k in range(8):
                            id0 = vertex2index[layer][self.hexs[layer][i][j]].item()
                            id1 = vertex2index[layer][self.hexs[layer][i][k]].item()
                            key = tuple([id0,id1])
                            if key not in my_dict:
                                my_dict[key] = 0
                sorted_dict = dict(sorted(my_dict.items(), key=operator.itemgetter(0)))
                self.UtAUs_nnz[layer-1] = len(my_dict)
                self.UtAUs_D_nnz[layer-1] = self.dims[layer]
                self.UtAUs_L_nnz[layer-1] = (len(my_dict)-self.dims[layer])>>1
                self.UtAUs_U_nnz[layer-1] = self.UtAUs_L_nnz[layer-1]
                self.UtAUs_off_l[layer-1] = 0
                self.UtAUs_off_d[layer-1] = self.UtAUs_L_nnz[layer-1]
                self.UtAUs_off_u[layer-1] = self.UtAUs_L_nnz[layer-1]+self.UtAUs_D_nnz[layer-1]
                self.UtAUs_D_row.append(torch.zeros((self.UtAUs_D_nnz[layer-1]),dtype=torch.int32,requires_grad=False))
                self.UtAUs_D_col.append(torch.zeros((self.UtAUs_D_nnz[layer-1]),dtype=torch.int32,requires_grad=False))
                self.UtAUs_L_row.append(torch.zeros((self.UtAUs_L_nnz[layer-1]),dtype=torch.int32,requires_grad=False))
                self.UtAUs_L_col.append(torch.zeros((self.UtAUs_L_nnz[layer-1]),dtype=torch.int32,requires_grad=False))
                self.UtAUs_U_row.append(torch.zeros((self.UtAUs_U_nnz[layer-1]),dtype=torch.int32,requires_grad=False))
                self.UtAUs_U_col.append(torch.zeros((self.UtAUs_U_nnz[layer-1]),dtype=torch.int32,requires_grad=False))

                self.UtAUs_GS_U_col.append(torch.zeros((self.UtAUs_U_nnz[layer-1]),dtype=torch.int32,requires_grad=False))
                self.UtAUs_GS_U_row.append(torch.zeros((self.UtAUs_U_nnz[layer-1]),dtype=torch.int32,requires_grad=False))
                self.UtAUs_GS_L_col.append(torch.zeros((self.UtAUs_L_nnz[layer-1]),dtype=torch.int32,requires_grad=False))
                self.UtAUs_GS_L_row.append(torch.zeros((self.UtAUs_L_nnz[layer-1]),dtype=torch.int32,requires_grad=False))
                self.UtAUs_GS_U_Ptr.append([0 for _ in range(self.color_num[layer]+1)])
                self.UtAUs_GS_L_Ptr.append([0 for _ in range(self.color_num[layer]+1)])
                self.UtAUs_block_offset.append(torch.zeros((len(my_dict)),dtype=torch.int32,requires_grad=False))

                i_l = 0
                i_d = 0
                i_u = 0
                i_f = 0
                p_base_l = -1
                p_base_u = -1
                now_base_l = 0
                next_base_l = self.color_vertex_num[layer][1]
                r_base_u = -1
                c_base_u = 0

                for key in sorted_dict.keys():
                    r = key[0]
                    c = key[1]
                    if c == r:
                        self.UtAUs_D_row[layer-1][i_d] = r
                        self.UtAUs_D_col[layer-1][i_d] = c
                        sorted_dict[key] = r+self.UtAUs_off_d[layer-1]
                        self.UtAUs_block_offset[layer-1][i_d+self.UtAUs_off_d[layer-1]] = i_f
                        i_d += 1
                        i_f += 1
                    if r<c:
                        while(r>=c_base_u):
                            p_base_u+=1
                            self.UtAUs_GS_U_Ptr[layer-1][p_base_u] = i_u
                            r_base_u = self.color_vertex_num[layer][p_base_u]
                            c_base_u = self.color_vertex_num[layer][p_base_u+1]
                        self.UtAUs_U_row[layer-1][i_u] = r
                        self.UtAUs_U_col[layer-1][i_u] = c
                        self.UtAUs_GS_U_row[layer-1][i_u] = r-r_base_u
                        self.UtAUs_GS_U_col[layer-1][i_u] = c-c_base_u 
                        self.UtAUs_block_offset[layer-1][i_u+self.UtAUs_off_u[layer-1]] = i_f
                        sorted_dict[key] = i_u+self.UtAUs_off_u[layer-1]
                        i_u += 1
                        i_f += 1
                    if r > c:
                        while(r>=next_base_l):
                            p_base_l+=1
                            self.UtAUs_GS_L_Ptr[layer-1][p_base_l] = i_l
                            now_base_l = self.color_vertex_num[layer][p_base_l+1]
                            next_base_l = self.color_vertex_num[layer][p_base_l+2]
                        self.UtAUs_L_row[layer-1][i_l] = r
                        self.UtAUs_L_col[layer-1][i_l] = c
                        self.UtAUs_GS_L_row[layer-1][i_l] = r-now_base_l
                        self.UtAUs_GS_L_col[layer-1][i_l] = c
                        self.UtAUs_block_offset[layer-1][i_l+self.UtAUs_off_l[layer-1]] = i_f
                        sorted_dict[key] = i_l+self.UtAUs_off_l[layer-1]
                        i_l += 1
                        i_f += 1
                for i in range(self.hexs[layer].shape[0]):
                    for j in range(8):
                        for k in range(8):
                            id0 = vertex2index[layer][self.hexs[layer][i][j]].item()
                            id1 = vertex2index[layer][self.hexs[layer][i][k]].item()
                            key = tuple([id0,id1])
                            hex_update_offset[layer][i*64+j*8+k] = sorted_dict[key]
                while(p_base_l<self.color_num[layer]-1):
                    p_base_l+=1
                    self.UtAUs_GS_L_Ptr[layer-1][p_base_l] = i_l
                while(p_base_u<self.color_num[layer]-1):
                    p_base_u+=1
                    self.UtAUs_GS_U_Ptr[layer-1][p_base_u] = i_u        
        self.hex_update_offset_gpu = [wp.from_torch(hex_update_offset[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer)]
        

        # prepare
        shapeFuncGrad = torch.zeros((8,8,3),dtype=torch.float32)
        shapeFuncGradMat = torch.zeros((8,8,3,3,3),dtype=torch.float32)
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
        calShapeFuncGradMat(shapeFuncGrad,shapeFuncGradMat)

        #for general hex,we need patial X/patial epsilon on the 8 quadrature points . Often we use the inverse of the matrix,so we save the inverse.
        IM = torch.eye(3,dtype=torch.float32)

        self.x_cpu = wp.from_torch(x,dtype=wp.vec3)
        self.x_gpu_layer = [wp.zeros(self.dims[i],dtype=wp.vec3f,requires_grad=True,device='cuda:0') for i in range(self.layer)]
        self.dev_x_init = wp.zeros(self.dims[0],dtype=wp.vec3f,requires_grad=False,device='cuda:0')
        wp.copy(self.dev_x_init,self.x_cpu)
        wp.copy(self.x_gpu_layer[0],self.x_cpu)
        for i in range(1,self.layer):
            bsr_mv(self.Ut_noOrder[i-1],self.x_gpu_layer[i-1],self.x_gpu_layer[i],alpha=1.0,beta=0.0)
        self.x_lineSearch = wp.zeros_like(self.x_cpu,device='cuda:0')
        self.grad_gpu = wp.zeros_like(self.x_cpu,device='cuda:0')
        wp.copy(self.x_lineSearch,self.x_cpu)

        self.IM_gpu = wp.from_torch(IM.to('cuda:0'),dtype=wp.mat33f)
        self.hexagons_gpu = [wp.from_torch(self.hexs[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer)]#wp.from_torch(hexagons.to('cuda:0'))
        self.shapeFuncGrad_gpu = wp.from_torch(shapeFuncGrad.to('cuda:0'),dtype=wp.vec3f)
        self.shapeFuncGradMat_gpu = wp.from_torch(shapeFuncGradMat.to('cuda:0'),dtype=wp.mat33f)
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

        self.inverse_pX_peps_gpu = [wp.array(shape = (self.hexs[i].shape[0],8),dtype=wp.mat33f,requires_grad=True) for i in range(self.layer)]
        self.det_pX_peps_gpu = [wp.array(shape = (self.hexs[i].shape[0],8),dtype=wp.float32,requires_grad=True) for i in range(self.layer)]
        self.dev_mat_F = [wp.array(shape = (self.hexs[i].shape[0]*8),dtype=wp.mat33f,requires_grad=True) for i in range(self.layer)]
        self.dev_mat_F_ = [wp.array(shape = (self.hexs[i].shape[0]*8),dtype=wp.mat33f,requires_grad=True) for i in range(self.layer)]
        self.dev_mat_E = [wp.array(shape = (self.hexs[i].shape[0]*8),dtype=wp.mat33f,requires_grad=True) for i in range(self.layer)]
        self.dev_mat_E_ = [wp.array(shape = (self.hexs[i].shape[0]*8),dtype=wp.mat33f,requires_grad=True) for i in range(self.layer)]

        for i in range(1):
            wp.launch(kernel=prepare_kernal,dim=self.hexs[i].shape[0]*8,inputs=[self.x_gpu_layer[i],self.hexagons_gpu[i],self.shapeFuncGrad_gpu,self.dev_mat_F[0],self.det_pX_peps_gpu[i],self.inverse_pX_peps_gpu[i]])
            wp.launch(kernel=Sum_8,dim = self.hexs[i].shape[0],inputs = [self.vol[i],self.det_pX_peps_gpu[i]])
            wp.launch(kernel=prepare_mass,dim=self.hexs[i].shape[0]*8,inputs=[self.vol[i],self.m_gpu[i],self.hexagons_gpu[i]])
        # pinned components
        self.MF_value_cpu = torch.zeros((self.MF_nnz,3,3),dtype=torch.float32,requires_grad=False)
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
            x[pinList[0]][1] += 0.001
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
        for i in range(self.N_pin):
            self.MF_value_cpu[vertex2index[0][pin_list[i]]+self.off_d] = torch.eye(3,dtype=torch.float32)*self.control_mag           
        self.MF_value_fixed_gpu = wp.from_torch(self.MF_value_cpu.to('cuda:0'),dtype=wp.mat33f)
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
        wp.copy(self.X,self.x_cpu)
        self.dev_E = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_R = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_P = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_AP = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_B = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_B_fixed = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_delta_x = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0',requires_grad=True) for i in range(self.layer)]
        self.dev_x_solved = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0',requires_grad=True) for i in range(self.layer)]
        #adam的系数
        self.m = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.v = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.m_hat = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.v_hat = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')


        self.MF_value_gpu = wp.zeros((self.MF_nnz),dtype=wp.mat33f,device='cuda:0')
        self.dev_temp_X = wp.zeros(shape=(self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.squared_sum = wp.zeros(shape=(1),dtype=wp.float32,device='cuda:0')
        self.dot_sum = wp.zeros(shape=(1),dtype=wp.float32,device='cuda:0')
        self.norm_max = wp.zeros(shape=(1),dtype=wp.float32,device='cuda:0')

        self.UtAUs_D_row_gpu = [wp.from_torch(self.UtAUs_D_row[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_D_col_gpu = [wp.from_torch(self.UtAUs_D_col[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_L_row_gpu = [wp.from_torch(self.UtAUs_L_row[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_L_col_gpu = [wp.from_torch(self.UtAUs_L_col[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]    
        self.UtAUs_U_row_gpu = [wp.from_torch(self.UtAUs_U_row[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_U_col_gpu = [wp.from_torch(self.UtAUs_U_col[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_row_gpu = [wp.zeros(shape=(self.UtAUs_nnz[i]),dtype=wp.int32,device='cuda:0') for i in range(self.layer-1)]
        self.UtAUs_col_gpu = [wp.zeros(shape=(self.UtAUs_nnz[i]),dtype=wp.int32,device='cuda:0') for i in range(self.layer-1)]
        self.UtAUs_GS_U_row_gpu = [wp.from_torch(self.UtAUs_GS_U_row[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_GS_U_col_gpu = [wp.from_torch(self.UtAUs_GS_U_col[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_GS_L_row_gpu = [wp.from_torch(self.UtAUs_GS_L_row[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_GS_L_col_gpu = [wp.from_torch(self.UtAUs_GS_L_col[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]

        self.UtAUs_GS_U = []
        self.UtAUs_GS_L = []
        for layer in range(self.layer-1):
            wp.copy(self.UtAUs_row_gpu[layer],self.UtAUs_D_row_gpu[layer],self.UtAUs_off_d[layer],0,self.UtAUs_D_nnz[layer])
            wp.copy(self.UtAUs_col_gpu[layer],self.UtAUs_D_col_gpu[layer],self.UtAUs_off_d[layer],0,self.UtAUs_D_nnz[layer])
            wp.copy(self.UtAUs_row_gpu[layer],self.UtAUs_L_row_gpu[layer],self.UtAUs_off_l[layer],0,self.UtAUs_L_nnz[layer])
            wp.copy(self.UtAUs_col_gpu[layer],self.UtAUs_L_col_gpu[layer],self.UtAUs_off_l[layer],0,self.UtAUs_L_nnz[layer])
            wp.copy(self.UtAUs_row_gpu[layer],self.UtAUs_U_row_gpu[layer],self.UtAUs_off_u[layer],0,self.UtAUs_U_nnz[layer])
            wp.copy(self.UtAUs_col_gpu[layer],self.UtAUs_U_col_gpu[layer],self.UtAUs_off_u[layer],0,self.UtAUs_U_nnz[layer])
            self.UtAUs_GS_U.append([bsr_zeros(self.color_vertex_num[layer+1][i+1]-self.color_vertex_num[layer+1][i],self.color_vertex_num[layer+1][self.color_num[layer+1]]-self.color_vertex_num[layer+1][i+1],wp.mat33f,device='cuda:0') for i in range(self.color_num[layer+1]-1)])
            self.UtAUs_GS_L.append([bsr_zeros(self.color_vertex_num[layer+1][i+2]-self.color_vertex_num[layer+1][i+1],self.color_vertex_num[layer+1][i+1],wp.mat33f,device='cuda:0') for i in range(self.color_num[layer+1]-1)])


        self.UtAUs_block_offset_gpu = [wp.from_torch(self.UtAUs_block_offset[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_values = [wp.zeros(shape=(self.UtAUs_nnz[i]),dtype=wp.mat33f,device='cuda:0',requires_grad=True) for i in range(self.layer-1)]
            
        self.plot_x = []
        self.plot_y = []
        self.plot_InfNorm = []
        self.plot_energy = []
        self.plot_InfNorm_newton = []
        self.plot_energy_newton = []
        self.plot_InfNorm_newtonMultigrid = []
        self.plot_energy_newtonMultigrid = []
        print('init done')



    def PerformJacobi(self,layer = 0,iterations = 2):
        self.dev_delta_x[layer].zero_()
        if layer == 0:
            for iter in range(iterations):
                wp.copy(self.dev_B[layer],self.dev_B_fixed[layer])
                bsr_mv(self.L[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
                bsr_mv(self.U[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
                self.dev_delta_x[layer].zero_()
                wp.launch(kernel=jacobi_iteration,dim=self.N_verts,inputs=[self.dev_delta_x[layer],self.MF_value_gpu,self.dev_B[layer],self.off_d])
        else:
            for iter in range(iterations):
                wp.copy(self.dev_B[layer],self.dev_B_fixed[layer])
                bsr_mv(self.L[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
                bsr_mv(self.U[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
                self.dev_delta_x[layer].zero_()
                wp.launch(kernel=jacobi_iteration,dim=self.dims[layer],inputs=[self.dev_delta_x[layer],self.D[layer].values,self.dev_B[layer],0])

    def PerformGaussSeidel(self,layer = 0,iterations = 10):
        self.dev_delta_x[layer].zero_()
        if layer == 0:
            for iter in range(iterations):
                wp.copy(self.dev_B[layer],self.dev_B_fixed[layer])
                bsr_mv(self.L[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
                self.dev_delta_x[layer].zero_()
                for color in range(self.color_num[layer]-1,-1,-1):
                    base = self.color_vertex_num[layer][color]
                    wp.launch(kernel=Colored_GS_MF_Kernel,dim=self.color_vertex_num[layer][color+1]-base,inputs=[self.dev_delta_x[layer],self.MF_value_gpu,self.dev_B[layer],base,self.color_vertex_num[layer][color+1]-base,self.off_d])
                    if color:
                        bsr_set_from_triplets(self.MF_GS_U[color-1],nnz = self.MF_GS_U_Ptr[color]-self.MF_GS_U_Ptr[color-1],rows=self.MF_GS_U_row_gpu,columns=self.MF_GS_U_col_gpu,values=self.MF_value_gpu,row_offset=self.MF_GS_U_Ptr[color-1],col_offset=self.MF_GS_U_Ptr[color-1],value_offset=self.off_u+self.MF_GS_U_Ptr[color-1])
                        bsr_mv(self.MF_GS_U[color-1],self.dev_delta_x[layer],self.dev_B[layer],x_offset=base,y_offset=self.color_vertex_num[layer][color-1],alpha=-1.0,beta=1.0)
                
                wp.copy(self.dev_B[layer],self.dev_B_fixed[layer])
                bsr_mv(self.U[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
                self.dev_delta_x[layer].zero_()
                for color in range(self.color_num[layer]):
                    base = self.color_vertex_num[layer][color]
                    wp.launch(kernel=Colored_GS_MF_Kernel,dim=self.color_vertex_num[layer][color+1]-base,inputs=[self.dev_delta_x[layer],self.MF_value_gpu,self.dev_B[layer],base,self.color_vertex_num[layer][color+1]-base,self.off_d])
                    if color < self.color_num[layer]-1:
                        bsr_set_from_triplets(self.MF_GS_L[color],nnz = self.MF_GS_L_Ptr[color+1]-self.MF_GS_L_Ptr[color],rows=self.MF_GS_L_row_gpu,columns=self.MF_GS_L_col_gpu,values=self.MF_value_gpu,row_offset=self.MF_GS_L_Ptr[color],col_offset=self.MF_GS_L_Ptr[color],value_offset=self.off_l+self.MF_GS_L_Ptr[color])
                        bsr_mv(self.MF_GS_L[color],self.dev_delta_x[layer],self.dev_B[layer],x_offset=0,y_offset=self.color_vertex_num[layer][color+1],alpha=-1.0,beta=1.0)
        else:
            for iter in range(iterations):
                wp.copy(self.dev_B[layer],self.dev_B_fixed[layer])
                bsr_mv(self.L[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
                self.dev_delta_x[layer].zero_()
                for color in range(self.color_num[layer]-1,-1,-1):
                    base = self.color_vertex_num[layer][color]
                    wp.launch(kernel=Colored_GS_MF_Kernel,dim=self.color_vertex_num[layer][color+1]-base,inputs=[self.dev_delta_x[layer],self.UtAUs_values[layer-1],self.dev_B[layer],base,self.color_vertex_num[layer][color+1]-base,self.UtAUs_off_d[layer-1]])
                    if color:
                        bsr_set_from_triplets(self.UtAUs_GS_U[layer-1][color-1],nnz = self.UtAUs_GS_U_Ptr[layer-1][color]-self.UtAUs_GS_U_Ptr[layer-1][color-1],rows=self.UtAUs_GS_U_row_gpu[layer-1],columns=self.UtAUs_GS_U_col_gpu[layer-1],values=self.UtAUs_values[layer-1],row_offset=self.UtAUs_GS_U_Ptr[layer-1][color-1],col_offset=self.UtAUs_GS_U_Ptr[layer-1][color-1],value_offset=self.UtAUs_off_u[layer-1]+self.UtAUs_GS_U_Ptr[layer-1][color-1])
                        bsr_mv(self.UtAUs_GS_U[layer-1][color-1],self.dev_delta_x[layer],self.dev_B[layer],x_offset=base,y_offset=self.color_vertex_num[layer][color-1],alpha=-1.0,beta=1.0)
                
                wp.copy(self.dev_B[layer],self.dev_B_fixed[layer])
                bsr_mv(self.U[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
                self.dev_delta_x[layer].zero_()
                for color in range(self.color_num[layer]):
                    base = self.color_vertex_num[layer][color]
                    wp.launch(kernel=Colored_GS_MF_Kernel,dim=self.color_vertex_num[layer][color+1]-base,inputs=[self.dev_delta_x[layer],self.UtAUs_values[layer-1],self.dev_B[layer],base,self.color_vertex_num[layer][color+1]-base,self.UtAUs_off_d[layer-1]])
                    if color < self.color_num[layer]-1:
                        bsr_set_from_triplets(self.UtAUs_GS_L[layer-1][color],nnz = self.UtAUs_GS_L_Ptr[layer-1][color+1]-self.UtAUs_GS_L_Ptr[layer-1][color],rows=self.UtAUs_GS_L_row_gpu[layer-1],columns=self.UtAUs_GS_L_col_gpu[layer-1],values=self.UtAUs_values[layer-1],row_offset=self.UtAUs_GS_L_Ptr[layer-1][color],col_offset=self.UtAUs_GS_L_Ptr[layer-1][color],value_offset=self.UtAUs_off_l[layer-1]+self.UtAUs_GS_L_Ptr[layer-1][color])
                        bsr_mv(self.UtAUs_GS_L[layer-1][color],self.dev_delta_x[layer],self.dev_B[layer],x_offset=0,y_offset=self.color_vertex_num[layer][color+1],alpha=-1.0,beta=1.0)

    def PerformConjugateGradient(self,layer = 0,iterations = 10,tol = 1e-5):
        self.dev_delta_x[layer].zero_()
        r0 = 0.0
        r1 = 0.0
        dot = 0.0
        alpha = 0.0
        beta = 0.0
        neg_alpha = 0.0
        wp.copy(self.dev_B[layer],self.dev_B_fixed[layer])
        self.squared_sum.zero_()
        wp.launch(kernel = square_sum,dim=self.dims[layer],inputs=[self.dev_B[layer],self.squared_sum])
        r1 = self.squared_sum.numpy()[0]
        
        if r1 < EPSILON:
            return
        r = r1
        k = 1
        while(r1>tol*r and k<=iterations):
            if k > 1:
                beta = r1/r0
                wp.launch(scal,dim=self.dims[layer],inputs=[self.dev_P[layer],beta])
                wp.launch(axpy,dim=self.dims[layer],inputs=[self.dev_P[layer],self.dev_B[layer],1.0])
            else :
                wp.copy(self.dev_P[layer],self.dev_B[layer])
            self.dev_AP[layer].zero_()
            bsr_mv(self.L[layer],self.dev_P[layer],self.dev_AP[layer],alpha=1.0,beta=1.0)
            bsr_mv(self.U[layer],self.dev_P[layer],self.dev_AP[layer],alpha=1.0,beta=1.0)
            bsr_mv(self.D[layer],self.dev_P[layer],self.dev_AP[layer],alpha=1.0,beta=1.0)
            self.dot_sum.zero_()
            wp.launch(kernel = cublasSdot,dim=self.dims[layer],inputs=[self.dev_P[layer],self.dev_AP[layer],self.dot_sum])
            dot = self.dot_sum.numpy()[0]
            if dot<1e-10:
                break
            alpha = r1/dot
            neg_alpha = -alpha
            wp.launch(axpy,dim=self.dims[layer],inputs=[self.dev_delta_x[layer],self.dev_P[layer],alpha])
            wp.launch(axpy,dim=self.dims[layer],inputs=[self.dev_B[layer],self.dev_AP[layer],neg_alpha])
            r0 = r1
            self.squared_sum.zero_()
            wp.launch(kernel = square_sum,dim=self.dims[layer],inputs=[self.dev_B[layer],self.squared_sum])
            r1 = self.squared_sum.numpy()[0]
            
            k+=1

    def PerformConjugateGradient_WithInitX(self,layer = 0,iterations = 10,tol = 1e-5):
        r0 = 0.0
        r1 = 0.0
        dot = 0.0
        alpha = 0.0
        beta = 0.0
        neg_alpha = 0.0
        wp.copy(self.dev_B[layer],self.dev_B_fixed[layer])
        bsr_mv(self.L[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.U[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.D[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
        self.squared_sum.zero_()
        wp.launch(kernel = square_sum,dim=self.dims[layer],inputs=[self.dev_B[layer],self.squared_sum])
        r1 = self.squared_sum.numpy()[0]
        
        if r1 < EPSILON:
            return
        r = r1
        k = 1
        while(r1>tol*r and k<=iterations):
            if k > 1:
                beta = r1/r0
                wp.launch(scal,dim=self.dims[layer],inputs=[self.dev_P[layer],beta])
                wp.launch(axpy,dim=self.dims[layer],inputs=[self.dev_P[layer],self.dev_B[layer],1.0])
            else :
                wp.copy(self.dev_P[layer],self.dev_B[layer])
            self.dev_AP[layer].zero_()
            bsr_mv(self.L[layer],self.dev_P[layer],self.dev_AP[layer],alpha=1.0,beta=1.0)
            bsr_mv(self.U[layer],self.dev_P[layer],self.dev_AP[layer],alpha=1.0,beta=1.0)
            bsr_mv(self.D[layer],self.dev_P[layer],self.dev_AP[layer],alpha=1.0,beta=1.0)
            self.dot_sum.zero_()
            wp.launch(kernel = cublasSdot,dim=self.dims[layer],inputs=[self.dev_P[layer],self.dev_AP[layer],self.dot_sum])
            dot = self.dot_sum.numpy()[0]
            if dot<1e-10:
                break
            alpha = r1/dot
            neg_alpha = -alpha
            wp.launch(axpy,dim=self.dims[layer],inputs=[self.dev_delta_x[layer],self.dev_P[layer],alpha])
            wp.launch(axpy,dim=self.dims[layer],inputs=[self.dev_B[layer],self.dev_AP[layer],neg_alpha])
            r0 = r1
            self.squared_sum.zero_()
            wp.launch(kernel = square_sum,dim=self.dims[layer],inputs=[self.dev_B[layer],self.squared_sum])
            r1 = self.squared_sum.numpy()[0]
            
            k+=1

    def downSample(self,layer = 0):
        wp.copy(self.dev_R[layer],self.dev_B_fixed[layer])
        wp.copy(self.dev_x_solved[layer],self.dev_delta_x[layer])
        bsr_mv(self.L[layer],self.dev_delta_x[layer],self.dev_R[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.U[layer],self.dev_delta_x[layer],self.dev_R[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.D[layer],self.dev_delta_x[layer],self.dev_R[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.Ut_hat[layer],self.dev_R[layer],self.dev_B_fixed[layer+1],alpha=1.0,beta=0.0)
    
    def upSample(self,layer = 1):
        #print(self.dev_delta_x[layer])
        wp.launch(kernel=addVec3,dim=self.dims[layer],inputs=[self.dev_x_solved[layer],self.dev_delta_x[layer]])
        bsr_mv(self.Us_hat[layer-1],self.dev_x_solved[layer],self.dev_x_solved[layer-1],alpha=1.0,beta=1.0)
        bsr_mv(self.L[layer-1],self.dev_x_solved[layer-1],self.dev_B_fixed[layer-1],alpha=-1.0,beta=1.0)
        bsr_mv(self.D[layer-1],self.dev_x_solved[layer-1],self.dev_B_fixed[layer-1],alpha=-1.0,beta=1.0)
        bsr_mv(self.U[layer-1],self.dev_x_solved[layer-1],self.dev_B_fixed[layer-1],alpha=-1.0,beta=1.0)
        
    def finish(self):
        wp.launch(kernel=addVec3,dim=self.dims[0],inputs=[self.dev_x_solved[0],self.dev_delta_x[0]])
        for i in range(1,self.layer):
            self.dev_x_solved[i].zero_()

    def showErrorInfNorm(self,layer:int,x:wp.array(dtype=wp.vec3)):
        #print('layer : ',layer)
        self.norm_max.zero_()
        wp.launch(kernel = Inf_norm,dim=self.dims[layer],inputs=[x,self.norm_max,self.dev_index2vertex[0],self.pin_gpu])
        print('Inf_norm : ',self.norm_max.numpy()[0])

    def showError(self,layer):
        print('layer : ',layer)
        wp.copy(self.dev_B[layer],self.dev_B_fixed[layer])
        # self.squared_sum.zero_()
        # wp.launch(kernel=square_sum,dim=self.dims[layer],inputs=[self.dev_B[layer],self.squared_sum])
        # print('before solve squared_norm : ',self.squared_sum.numpy()[0])
        self.norm_max.zero_()
        wp.launch(kernel = Inf_norm,dim=self.dims[layer],inputs=[self.dev_B[layer],self.norm_max,self.dev_index2vertex[0],self.pin_gpu])
        print('before solve Inf_norm : ',self.norm_max.numpy()[0])
        bsr_mv(self.L[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.U[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.D[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
        # self.squared_sum.zero_()
        # wp.launch(kernel=square_sum,dim=self.dims[layer],inputs=[self.dev_B[layer],self.squared_sum])
        # print('after solve  squared_norm : ',self.squared_sum.numpy()[0])
        self.norm_max.zero_()
        wp.launch(kernel = Inf_norm,dim=self.dims[layer],inputs=[self.dev_B[layer],self.norm_max,self.dev_index2vertex[0],self.pin_gpu])
        print('after solve Inf_norm : ',self.norm_max.numpy()[0])


    # 参考 https://www.math.hkust.edu.hk/~mamu/courses/531/tutorial_with_corrections.pdf   98页的公式
    def FASv0(self,iterations = 100):
        wp.copy(self.dev_x_solved[0],self.x_gpu_layer[0])
        for step in range(1,iterations+1):
            self.dev_B_fixed[0].zero_()
            self.dev_B_fixed[1].zero_()
            self.MF_value_gpu.zero_()
            self.UtAUs_values[0].zero_()
            
            wp.launch(kernel=compute_elastic_hessian,dim=self.hexs[0].shape[0]*64,inputs=[self.dev_x_solved[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.MF_value_gpu,self.hex_update_offset_gpu[0]])
            wp.launch(kernel=spd_matrix33f,dim=self.MF_nnz,inputs=[self.MF_value_gpu,self.spd_value])
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.dev_x_solved[0],self.hexagons_gpu[0],self.dev_vertex2index[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.dev_B_fixed[0]])
            #bsr_set_from_triplets(self.A[0],self.MF_row_gpu,self.MF_col_gpu,self.MF_value_gpu)
            bsr_set_from_triplets(self.L[0],self.MF_L_row_gpu,self.MF_L_col_gpu,self.MF_value_gpu,value_offset=self.off_l)
            bsr_set_from_triplets(self.U[0],self.MF_U_row_gpu,self.MF_U_col_gpu,self.MF_value_gpu,value_offset=self.off_u)
            bsr_set_from_triplets(self.D[0],self.MF_D_row_gpu,self.MF_D_col_gpu,self.MF_value_gpu,value_offset=self.off_d)
            #先在fine 网格上进行GS迭代
            self.PerformGaussSeidel(layer=0,iterations=1)
            self.showError(layer = 0)   
   

            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.dev_x_solved[0],self.dev_delta_x[0],self.dev_index2vertex[0]])
            #Restrict the current approximation and its fine-grid residual to the coarse grid
            bsr_mv(self.Ut_noOrder[0],self.dev_x_solved[0],self.dev_x_solved[1],alpha=1.0,beta=0.0)
            #self.x[1] = warp.to_torch(self.dev_x_solved[1]).cpu()
            wp.copy(self.dev_R[0],self.dev_B_fixed[0])
            self.dev_R[0].zero_()
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.dev_x_solved[0],self.hexagons_gpu[0],self.dev_vertex2index[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.dev_R[0]])
            # # wp.launch(kernel=z_axpby,dim = self.dims[0],inputs = [self.dev_R[0],self.dev_B_fixed[0],self.dev_B[0],-1.0,1.0])
            # bsr_mv(self.L[0],self.dev_delta_x[0],self.dev_R[0],alpha=-1.0,beta=1.0)
            # bsr_mv(self.U[0],self.dev_delta_x[0],self.dev_R[0],alpha=-1.0,beta=1.0)
            # bsr_mv(self.D[0],self.dev_delta_x[0],self.dev_R[0],alpha=-1.0,beta=1.0)

            bsr_mv(self.Ut_hat[0],self.dev_R[0],self.dev_B_fixed[1],alpha=1.0,beta=0.0)
            #Solve the coarse-grid problem
            #wp.launch(kernel=axpby,dim=self.dims[1],inputs=[self.dev_R[1],self.dev_B_fixed[1],1.0,-1.0])
            #wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.hexs[1].shape[0]*8,inputs=[self.dev_x_solved[1],self.hexagons_gpu[1],self.dev_vertex2index[1],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.dev_B_fixed[1]])
            wp.launch(kernel=compute_elastic_hessian,dim=self.hexs[1].shape[0]*64,inputs=[self.dev_x_solved[1],self.hexagons_gpu[1],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.UtAUs_values[0],self.hex_update_offset_gpu[1]])
            wp.launch(kernel=spd_matrix33f,dim=self.UtAUs_nnz[0],inputs=[self.UtAUs_values[0],self.spd_value])
            bsr_set_from_triplets(self.L[1],self.UtAUs_L_row_gpu[0],self.UtAUs_L_col_gpu[0],self.UtAUs_values[0],value_offset=self.UtAUs_off_l[0])
            bsr_set_from_triplets(self.U[1],self.UtAUs_U_row_gpu[0],self.UtAUs_U_col_gpu[0],self.UtAUs_values[0],value_offset=self.UtAUs_off_u[0])
            bsr_set_from_triplets(self.D[1],self.UtAUs_D_row_gpu[0],self.UtAUs_D_col_gpu[0],self.UtAUs_values[0],value_offset=self.UtAUs_off_d[0])
            #设置cg初值
            wp.launch(kernel= V2I,dim = self.dims[1],inputs = [self.dev_delta_x[1],self.dev_x_solved[1],self.dev_vertex2index[1]])
            bsr_mv(self.L[1],self.dev_delta_x[1],self.dev_B_fixed[1],alpha=1.0,beta=1.0)
            bsr_mv(self.U[1],self.dev_delta_x[1],self.dev_B_fixed[1],alpha=1.0,beta=1.0)
            bsr_mv(self.D[1],self.dev_delta_x[1],self.dev_B_fixed[1],alpha=1.0,beta=1.0)
            #wp.launch(kernel=axpby,dim=self.dims[1],inputs=[self.dev_R[1],self.dev_B_fixed[1],1.0,1.0])
            #wp.copy(self.dev_delta_x[1],self.dev_x_solved[1])
            self.PerformConjugateGradient_WithInitX(layer = 1,iterations=10)
            wp.launch(kernel= I2V,dim = self.dims[1],inputs = [self.dev_B[1],self.dev_delta_x[1],self.dev_index2vertex[1]])
            self.x[1] = warp.to_torch(self.dev_B[1]).cpu()
            #self.show_layer(1)
            #Compute the coarse-grid approximation to the error
            wp.launch(kernel=z_axpby,dim=self.dims[1],inputs=[self.dev_E[1],self.dev_x_solved[1],self.dev_B[1],-1.0,1.0])
            #Interpolate the error approximation up to the fine grid and correct the current fine-grid approximation
            bsr_mv(self.Us_noOrder_hat[0],self.dev_E[1],self.dev_E[0],alpha=1.0,beta=0.0)
            wp.launch(kernel=axpby,dim=self.dims[0],inputs=[self.dev_E[0],self.dev_x_solved[0],1.0,1.0])
            #self.PerformJacobi(layer=1,iterations=1)
            self.showError(layer = 1)
            # bsr_mv(self.Us_hat[0],self.dev_delta_x[1],self.dev_delta_x[0],alpha=1.0,beta=0.0)
            # wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.dev_x_solved[0],self.dev_delta_x[0],self.dev_index2vertex[0]])
            self.x[0] = warp.to_torch(self.dev_x_solved[0]).cpu()
            # self.show_layer(0)

    def FASv1(self,iterations = 100):
        wp.copy(self.dev_x_solved[0],self.x_gpu_layer[0])
        for step in range(1,iterations+1):
            self.dev_B_fixed[0].zero_()
            self.dev_B_fixed[1].zero_()
            self.MF_value_gpu.zero_()
            self.UtAUs_values[0].zero_()
            
            wp.launch(kernel=compute_elastic_hessian,dim=self.hexs[0].shape[0]*64,inputs=[self.dev_x_solved[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.MF_value_gpu,self.hex_update_offset_gpu[0]])
            wp.launch(kernel=spd_matrix33f,dim=self.MF_nnz,inputs=[self.MF_value_gpu,self.spd_value])
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.dev_x_solved[0],self.hexagons_gpu[0],self.dev_vertex2index[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.dev_B_fixed[0]])
            #bsr_set_from_triplets(self.A[0],self.MF_row_gpu,self.MF_col_gpu,self.MF_value_gpu)
            bsr_set_from_triplets(self.L[0],self.MF_L_row_gpu,self.MF_L_col_gpu,self.MF_value_gpu,value_offset=self.off_l)
            bsr_set_from_triplets(self.U[0],self.MF_U_row_gpu,self.MF_U_col_gpu,self.MF_value_gpu,value_offset=self.off_u)
            bsr_set_from_triplets(self.D[0],self.MF_D_row_gpu,self.MF_D_col_gpu,self.MF_value_gpu,value_offset=self.off_d)
            #先在fine 网格上进行GS迭代
            self.PerformGaussSeidel(layer=0,iterations=1)
            self.showError(layer = 0)  
   

            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.dev_x_solved[0],self.dev_delta_x[0],self.dev_index2vertex[0]])
            #Restrict the current approximation and its fine-grid residual to the coarse grid
            bsr_mv(self.Ut_noOrder[0],self.dev_x_solved[0],self.dev_x_solved[1],alpha=1.0,beta=0.0)
            #self.x[1] = warp.to_torch(self.dev_x_solved[1]).cpu()
            wp.copy(self.dev_R[0],self.dev_B_fixed[0])
            self.dev_R[0].zero_()
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.dev_x_solved[0],self.hexagons_gpu[0],self.dev_vertex2index[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.dev_R[0]])

            bsr_mv(self.Ut_hat[0],self.dev_R[0],self.dev_B_fixed[1],alpha=1.0,beta=0.0)
            #Solve the coarse-grid problem
            #wp.launch(kernel=axpby,dim=self.dims[1],inputs=[self.dev_R[1],self.dev_B_fixed[1],1.0,-1.0])
            #wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.hexs[1].shape[0]*8,inputs=[self.dev_x_solved[1],self.hexagons_gpu[1],self.dev_vertex2index[1],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.dev_B_fixed[1]])
            wp.launch(kernel=compute_elastic_hessian,dim=self.hexs[1].shape[0]*64,inputs=[self.dev_x_solved[1],self.hexagons_gpu[1],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.UtAUs_values[0],self.hex_update_offset_gpu[1]])
            wp.launch(kernel=spd_matrix33f,dim=self.UtAUs_nnz[0],inputs=[self.UtAUs_values[0],self.spd_value])
            bsr_set_from_triplets(self.L[1],self.UtAUs_L_row_gpu[0],self.UtAUs_L_col_gpu[0],self.UtAUs_values[0],value_offset=self.UtAUs_off_l[0])
            bsr_set_from_triplets(self.U[1],self.UtAUs_U_row_gpu[0],self.UtAUs_U_col_gpu[0],self.UtAUs_values[0],value_offset=self.UtAUs_off_u[0])
            bsr_set_from_triplets(self.D[1],self.UtAUs_D_row_gpu[0],self.UtAUs_D_col_gpu[0],self.UtAUs_values[0],value_offset=self.UtAUs_off_d[0])

            self.PerformJacobi(layer=1,iterations=1)
            self.showError(layer = 1)
            bsr_mv(self.Us_hat[0],self.dev_delta_x[1],self.dev_delta_x[0],alpha=1.0,beta=0.0)
            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.dev_x_solved[0],self.dev_delta_x[0],self.dev_index2vertex[0]])
            self.x[0] = warp.to_torch(self.dev_x_solved[0]).cpu()
            # self.show_layer(0)

    def save_p_hat(self):
        temp = wp.to_torch(self.Ut_noOrder_value)
        torch.save(temp, 'assets/p/Ut_noOrder_0.pth')

    def train(self,iterations = 100):
        wp_l = wp.zeros((1),dtype=wp.float32,requires_grad=True)
        self.dev_B_fixed[0].zero_()
        self.dev_B_fixed[1].zero_()
        self.UtAUs_values[0].zero_()
        self.dev_mat_F[0].zero_()
        self.dev_mat_F[1].zero_()
        self.dev_mat_F_[0].zero_()
        self.dev_mat_F_[1].zero_()
        self.dev_delta_x[0].zero_()
        self.dev_delta_x[1].zero_()
        self.dev_x_solved[0].zero_()
        self.dev_R[0].zero_()
        wp.launch(kernel=compute_mat_F,dim = self.hexs[0].shape[0]*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.dev_mat_F_[0],self.dev_mat_E_[0]])
        wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.dev_vertex2index[0],self.dev_mat_F_[0],self.dev_mat_E_[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.dev_R[0]])
        bsr_mv(self.Ut_hat[0],self.dev_R[0],self.dev_B_fixed[1],alpha=1.0,beta=0.0)

        tape = wp.Tape()
        #self.Ut_noOrder_value = wp.from_torch(torch.load('assets/p_record/Ut_noOrder_0.pth').to('cuda:0'),requires_grad=True)

        for step in range(1,iterations+1):

            self.dev_B_fixed[0].zero_()
            self.dev_B_fixed[1].zero_()
            self.UtAUs_values[0].zero_()
            self.dev_mat_F[0].zero_()
            self.dev_mat_F[1].zero_()
            self.dev_mat_F_[0].zero_()
            self.dev_mat_F_[1].zero_()
            self.dev_delta_x[0].zero_()
            self.dev_delta_x[1].zero_()
            self.dev_x_solved[0].zero_()
            wp_l.zero_()
            #set x0
            wp.copy(self.x_cpu,self.dev_x_init)
            ri = self.surface_vert[torch.randint(0,len(self.surface_vert),size=(1,))]
            rand_val = (torch.rand(3)-0.5)*0.0002
            self.x[0][732] +=  rand_val
            #print('rand_val : ',rand_val)
            wp.copy(self.x_gpu_layer[0],self.x_cpu)

            self.dev_R[0].zero_()
            wp.launch(kernel=compute_mat_F,dim = self.hexs[0].shape[0]*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.dev_mat_F_[0],self.dev_mat_E_[0]])
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.dev_vertex2index[0],self.dev_mat_F_[0],self.dev_mat_E_[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.dev_R[0]])
            bsr_mv(self.Ut_hat[0],self.dev_R[0],self.dev_B_fixed[1],alpha=1.0,beta=0.0)

            tape.reset()
            with tape:
                wp.launch(kernel=Kronecker_product,dim=self.Ut_noOrder[0].nnz,inputs = [self.Ut_noOrder_value,self.IM_gpu,self.Ut_noOrder[0].values])
                bsr_mv(self.Ut_noOrder[0],self.dev_x_init,self.dev_x_solved[1],alpha=1.0,beta=0.0) 
                wp.launch(kernel=prepare_kernal,dim=self.hexs[1].shape[0]*8,inputs=[self.dev_x_solved[1],self.hexagons_gpu[1],self.shapeFuncGrad_gpu,self.dev_mat_F_[1],self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1]])
                bsr_mv(self.Ut_noOrder[0],self.X,self.x_gpu_layer[1],alpha=1.0,beta=0.0)      
                wp.launch(kernel=compute_mat_F,dim = self.hexs[1].shape[0]*8,inputs=[self.x_gpu_layer[1],self.hexagons_gpu[1],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1],self.IM_gpu,self.dev_mat_F[1],self.dev_mat_E[1]])
                wp.launch(kernel=compute_elastic_hessian,dim=self.hexs[1].shape[0]*64,inputs=[self.x_gpu_layer[1],self.hexagons_gpu[1],self.shapeFuncGradMat_gpu,self.shapeFuncGrad_gpu,self.dev_mat_F[1],self.dev_mat_E[1],self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.UtAUs_values[0],self.hex_update_offset_gpu[1]])                
                wp.launch(kernel=jacobi_iteration,dim=self.dims[1],inputs=[self.dev_delta_x[1],self.UtAUs_values[0],self.dev_B_fixed[1],self.UtAUs_off_d[0]])
                bsr_mv(self.Us_hat[0],self.dev_delta_x[1],self.dev_delta_x[0],alpha=1.0,beta=0.0)
                wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.dev_delta_x[0],self.dev_index2vertex[0]])
                wp.launch(kernel=compute_mat_F,dim = self.hexs[0].shape[0]*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.dev_mat_F[0],self.dev_mat_E[0]])
                wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.dev_vertex2index[0],self.dev_mat_F[0],self.dev_mat_E[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.dev_x_solved[0]])
                #wp.launch(kernel= loss_mat,dim = self.hexs[1].shape[0]*8,inputs=[self.inverse_pX_peps_gpu[1]],outputs=[wp_l])
                #wp.launch(kernel = loss_val,dim = self.hexs[1].shape[0]*8,inputs=[self.det_pX_peps_gpu[1]],outputs=[wp_l])
                wp.launch(kernel=loss,dim = self.dims[0],inputs=[self.dev_x_solved[0]],outputs=[wp_l])
            tape.zero()           
            tape.backward(loss=wp_l)  # compute gradients
            wp.launch(kernel=Valueaxpy,dim=self.Ut_noOrder[0].nnz,inputs=[self.Ut_noOrder_value,self.Ut_noOrder_value.grad,1e-4])
            # grad= wp.to_torch(self.Ut_noOrder_value.grad)
            # val = wp.to_torch(self.Ut_noOrder_value)
            # print(grad[5191],val[5191])
            # for i in range(grad.shape[0]):
            #     if grad[i] !=0:
            #        print(i)
            #        print(grad[i],val[i])
            #print(wp_l)
            if step%(iterations/100) == 0:
                print(step,'%')
        self.save_p_hat()
        self.x[1] = self.x_gpu_layer[1].numpy()
        self.show_layer(1)


    def compare(self,iterations = 100):
        #先用经典的方法
        for step in range(1,iterations+1):
            self.dev_R[0].zero_()
            self.dev_B_fixed[0].zero_()
            self.dev_B_fixed[1].zero_()
            self.UtAUs_values[0].zero_()
            self.dev_mat_F[0].zero_()
            self.dev_mat_F[1].zero_()
            self.dev_mat_F_[0].zero_()
            self.dev_mat_F_[1].zero_()
            self.dev_delta_x[0].zero_()
            self.dev_delta_x[1].zero_()
            self.dev_x_solved[0].zero_()
            wp.launch(kernel=compute_mat_F,dim = self.hexs[0].shape[0]*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.dev_mat_F_[0],self.dev_mat_E_[0]])
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.dev_vertex2index[0],self.dev_mat_F_[0],self.dev_mat_E_[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.dev_R[0]])
            self.plot_x.append(step)
            self.showErrorInfNorm(layer = 0,x = self.dev_R[0])
            self.plot_InfNorm.append(self.norm_max.numpy()[0])
            bsr_mv(self.Ut_hat[0],self.dev_R[0],self.dev_B_fixed[1],alpha=1.0,beta=0.0)

            wp.launch(kernel=Kronecker_product,dim=self.Ut_noOrder[0].nnz,inputs = [self.Ut_noOrder_value,self.IM_gpu,self.Ut_noOrder[0].values])
            bsr_mv(self.Ut_noOrder[0],self.dev_x_init,self.dev_x_solved[1],alpha=1.0,beta=0.0) 
            wp.launch(kernel=prepare_kernal,dim=self.hexs[1].shape[0]*8,inputs=[self.dev_x_solved[1],self.hexagons_gpu[1],self.shapeFuncGrad_gpu,self.dev_mat_F_[1],self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1]])
            bsr_mv(self.Ut_noOrder[0],self.X,self.x_gpu_layer[1],alpha=1.0,beta=0.0)      
            wp.launch(kernel=compute_mat_F,dim = self.hexs[1].shape[0]*8,inputs=[self.x_gpu_layer[1],self.hexagons_gpu[1],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1],self.IM_gpu,self.dev_mat_F[1],self.dev_mat_E[1]])
            wp.launch(kernel=compute_elastic_hessian,dim=self.hexs[1].shape[0]*64,inputs=[self.x_gpu_layer[1],self.hexagons_gpu[1],self.shapeFuncGradMat_gpu,self.shapeFuncGrad_gpu,self.dev_mat_F[1],self.dev_mat_E[1],self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.UtAUs_values[0],self.hex_update_offset_gpu[1]])                
            wp.launch(kernel=jacobi_iteration,dim=self.dims[1],inputs=[self.dev_delta_x[1],self.UtAUs_values[0],self.dev_B_fixed[1],self.UtAUs_off_d[0]])
            bsr_mv(self.Us_hat[0],self.dev_delta_x[1],self.dev_delta_x[0],alpha=1.0,beta=0.0)
            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.dev_delta_x[0],self.dev_index2vertex[0]])
            
        #再用训练完成的方法
        wp.copy(self.x_gpu_layer[0],self.x_cpu)
        self.Ut_noOrder_value=wp.from_torch(torch.load('assets/p/Ut_noOrder_0.pth').to('cuda:0'))
        for step in range(1,iterations+1):
            self.dev_R[0].zero_()
            self.dev_B_fixed[0].zero_()
            self.dev_B_fixed[1].zero_()
            self.UtAUs_values[0].zero_()
            self.dev_mat_F[0].zero_()
            self.dev_mat_F[1].zero_()
            self.dev_mat_F_[0].zero_()
            self.dev_mat_F_[1].zero_()
            self.dev_delta_x[0].zero_()
            self.dev_delta_x[1].zero_()
            self.dev_x_solved[0].zero_()
            wp.launch(kernel=compute_mat_F,dim = self.hexs[0].shape[0]*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.dev_mat_F_[0],self.dev_mat_E_[0]])
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.dev_vertex2index[0],self.dev_mat_F_[0],self.dev_mat_E_[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.dev_R[0]])
            self.showErrorInfNorm(layer = 0,x = self.dev_R[0])

            self.plot_InfNorm_newton.append(self.norm_max.numpy()[0])
            bsr_mv(self.Ut_hat[0],self.dev_R[0],self.dev_B_fixed[1],alpha=1.0,beta=0.0)

            wp.launch(kernel=Kronecker_product,dim=self.Ut_noOrder[0].nnz,inputs = [self.Ut_noOrder_value,self.IM_gpu,self.Ut_noOrder[0].values])
            bsr_mv(self.Ut_noOrder[0],self.dev_x_init,self.dev_x_solved[1],alpha=1.0,beta=0.0) 
            wp.launch(kernel=prepare_kernal,dim=self.hexs[1].shape[0]*8,inputs=[self.dev_x_solved[1],self.hexagons_gpu[1],self.shapeFuncGrad_gpu,self.dev_mat_F_[1],self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1]])
            bsr_mv(self.Ut_noOrder[0],self.X,self.x_gpu_layer[1],alpha=1.0,beta=0.0)      
            wp.launch(kernel=compute_mat_F,dim = self.hexs[1].shape[0]*8,inputs=[self.x_gpu_layer[1],self.hexagons_gpu[1],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1],self.IM_gpu,self.dev_mat_F[1],self.dev_mat_E[1]])
            wp.launch(kernel=compute_elastic_hessian,dim=self.hexs[1].shape[0]*64,inputs=[self.x_gpu_layer[1],self.hexagons_gpu[1],self.shapeFuncGradMat_gpu,self.shapeFuncGrad_gpu,self.dev_mat_F[1],self.dev_mat_E[1],self.det_pX_peps_gpu[1],self.inverse_pX_peps_gpu[1],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.UtAUs_values[0],self.hex_update_offset_gpu[1]])                
            wp.launch(kernel=jacobi_iteration,dim=self.dims[1],inputs=[self.dev_delta_x[1],self.UtAUs_values[0],self.dev_B_fixed[1],self.UtAUs_off_d[0]])
            bsr_mv(self.Us_hat[0],self.dev_delta_x[1],self.dev_delta_x[0],alpha=1.0,beta=0.0)
            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.dev_delta_x[0],self.dev_index2vertex[0]])
                
        #绘制图像
        plt.plot(self.plot_x,self.plot_InfNorm,linestyle='-', color='blue', label='classic U')
        plt.plot(self.plot_x,self.plot_InfNorm_newton,linestyle='-', color='red', label='trained U')

        plt.xlabel('iterations')
        plt.ylabel('norm')
        plt.legend()
        plt.show()
        return
    


    
    def show_layer(self,layer=0):
        index = np.array([0,4,6,2,1,5,7,3])
        cells = self.hexs[layer].numpy()[:,index]
        hex_num = np.zeros((cells.shape[0],1),dtype=np.int32)
        hex_num.fill(8)
        cells = np.concatenate((hex_num,cells),axis=1)
        cell_type = np.zeros(cells.shape[0],dtype=np.int8)
        cell_type.fill(pv.CellType.HEXAHEDRON)

        grid = pv.UnstructuredGrid(cells, cell_type, self.x[layer])
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
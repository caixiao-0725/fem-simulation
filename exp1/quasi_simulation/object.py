import pyvista as pv
import warp as wp
import warp.sparse
import warp.optim.linear
import torch            #warp只能在gpu上运行，所以这里用torch来做一些cpu上的操作再运到warp上
import numpy as np
from hexagon import *
from cublas import *
from cpu_function import *
import time
import sys
import operator

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
    MF_GS_U_rowInd_gpu : wp.array(dtype=wp.int32)
    MF_GS_U_colInd_gpu : wp.array(dtype=wp.int32)
    MF_GS_U_Ptr_gpu : wp.array(dtype=wp.int32)
    MF_GS_value_gpu : wp.array(dtype=wp.mat33f) 
    MF_GS_U_row_0_gpu : wp.array(dtype=wp.int32)
    MF_GS_U_col_0_gpu : wp.array(dtype=wp.int32)

    def __init__(self,mesh_path,dx,pinList):
        # 层数
        self.layer = 2
        

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
        # simulation components
        x = torch.tensor(self.voxels.points,dtype=torch.float32,requires_grad=False)
        hexagons = torch.zeros((self.N_hexagons,8),dtype=torch.int32,requires_grad=False)
        
        for i in range(self.N_hexagons):
            for j in range(8):
                hexagons[i][j] = hex[i][j]

        # fine to coarse
        self.dims = []
        self.x = []
        self.handles = []
        self.dims.append(self.N_verts)
        self.x.append(x)
        self.handles.append(hexagons)
        
        for layer in range(self.layer-1):
            coarse_num = 0
            #粗网格的网格大小是细网格的两倍
            if layer == 0:
                fine_size = self.dx
            else:
                fine_size = self.dx*2
            coarse_size = fine_size*2
            #计算细网格的bounding box的min
            min_ = torch.tensor([1000.0,1000.0,1000.0],dtype=torch.float32,requires_grad=False)
            for i in range(self.N_verts):
                for j in range(3):
                    if self.x[layer][i][j]<min_[j]:
                        min_[j] = x[i][j]
            #建立hash表 对于每一个立方体，算出它的中心对应的整数索引进行分类
            ijk_hash = {}
            mid_offset = fine_size*torch.ones((3),dtype=torch.float32,requires_grad=False)
            for i in range(self.N_hexagons):
                #计算中心 hex里面第一个点是最小的点，所以直接加上dx/2就是中心
                ijk = ijk_index(self.x[layer][hexagons[i][0]]+mid_offset,min_,coarse_size)
                key = tuple([ijk[0],ijk[1],ijk[2]])
                if key not in ijk_hash:
                    ijk_hash[key] = coarse_num
                    coarse_num += 1
            #建立新的hexagons 和 x   以及粗网格的hexagon包含的细网格的点的索引
            self.dims.append(coarse_num)
            new_x = torch.zeros((coarse_num,3),dtype=torch.float32,requires_grad=False)
            self.x.append(new_x)
            vertToCoarse = torch.zeros((self.N_verts,8),dtype=torch.int32,requires_grad=False)
            vertToCoarse.fill_(-1)
            vertToCoarse_num = torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False)
            for i in range(self.N_hexagons):
                ijk = ijk_index(self.x[layer][hexagons[i][0]]+mid_offset,min_,coarse_size)
                key = tuple([ijk[0],ijk[1],ijk[2]])
                coarse_id = ijk_hash[key]
                for j in range(8):
                    vert_id = hexagons[i][j]
                    temp_bool = True
                    for k in range(8):
                        if vertToCoarse[vert_id][k] == coarse_id:
                            temp_bool = False
                            break
                    if temp_bool:
                        vertToCoarse[vert_id][vertToCoarse_num[vert_id]] = coarse_id
                        vertToCoarse_num[vert_id] += 1


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

        vertex2index = torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False)
        index2vertex = torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False)

        #prepare for colored gauss seidel     edge_num  edge_to
        edge_num = np.zeros((self.N_verts),dtype=np.int32)
        edge_to = np.zeros((self.N_verts,6),dtype=np.int32)
        edge_per_hex = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],dtype=np.int32)

        edge_hash = {}

        for cell in self.voxels.cell:
            for j in range(12):
                id0 = cell.point_ids[edge_per_hex[j][0]]
                id1 = cell.point_ids[edge_per_hex[j][1]]
                if id0>id1:
                    key = tuple([id1,id0])
                else:
                    key = tuple([id0,id1])
                if key not in edge_hash:
                    edge_hash[key] = 0
                    edge_to[id0][edge_num[id0]] = id1
                    edge_num[id0] += 1
                    edge_to[id1][edge_num[id1]] = id0
                    edge_num[id1] += 1

        fixed = np.zeros((self.N_verts),dtype=np.bool_)
        fixed_num = self.N_verts
        color = torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False)

        init_index = 0
        color_list =[]
        color[init_index] = 0
        fixed[init_index] = True
        fixed_num -= 1
        while(fixed_num>0):
            for j in range(edge_num[init_index]):
                if fixed[edge_to[init_index][j]] == False:
                    fixed[edge_to[init_index][j]] = True
                    fixed_num -= 1
                    color[edge_to[init_index][j]] = color[init_index]^1
                    color_list.append(edge_to[init_index][j])
            init_index = color_list.pop(0)
        
        index_idx = 0
        self.color_vertex_num = [0,0,0]
        self.color_vertex_num[0] = 0
        for i in range(self.N_verts):
            if color[i] == 0:
                vertex2index[i] = index_idx
                index2vertex[index_idx] = i
                index_idx += 1
        self.color_vertex_num[1] = index_idx
        for i in range(self.N_verts):
            if color[i] == 1:
                vertex2index[i] = index_idx
                index2vertex[index_idx] = i
                index_idx += 1
        self.color_vertex_num[2] = index_idx
        self.color_num = 2
        

                        
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
                        id0 = vertex2index[hex[i][j]].item()
                        id1 = vertex2index[hex[i][k]].item()
                        key = tuple([id0,id1])
                        if key not in my_dict:
                            my_dict[key] = 0
            sorted_dict = dict(sorted(my_dict.items(), key=operator.itemgetter(0)))
            

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
            MF_GS_L_col = torch.zeros((self.MF_L_nnz),dtype=torch.int32,requires_grad=False)
            MF_GS_L_row = torch.zeros((self.MF_L_nnz),dtype=torch.int32,requires_grad=False)
            MF_GS_L_Ptr = torch.zeros((3),dtype=torch.int32,requires_grad=False)
            MF_GS_U_col = torch.zeros((self.MF_U_nnz),dtype=torch.int32,requires_grad=False)
            MF_GS_U_row = torch.zeros((self.MF_U_nnz),dtype=torch.int32,requires_grad=False)
            MF_GS_U_Ptr = torch.zeros((3),dtype=torch.int32,requires_grad=False)

            
            i_l = 0
            i_d = 0
            i_u = 0
            p_base_l = -1
            p_base_u = -1
            now_base_l = 0
            next_base_l = self.color_vertex_num[1]
            r_base_u = -1
            c_base_u = 0

            # for key in sorted_dict.keys():
            #     r = key[0]
            #     c = key[1]
            #     # if r<10:
            #     #     print(r,c)
            #     if c == r:
            #         self.MF_D_row[i_d] = r
            #         self.MF_D_col[i_d] = c
            #         sorted_dict[key] = r+self.off_d
            #         diag_offset[i_d] = r+self.off_d
            #         i_d += 1
            #     if r<c:
            #         while(r>=c_base_u):
            #             p_base_u+=1
            #             MF_GS_U_Ptr[p_base_u] = i_u
            #             r_base_u = self.color_vertex_num[p_base_u]
            #             c_base_u = self.color_vertex_num[p_base_u+1]
            #         self.MF_U_row[i_u] = r
            #         self.MF_U_col[i_u] = c
            #         MF_GS_U_row[i_u] = r-r_base_u
            #         MF_GS_U_col[i_u] = c-c_base_u 
            #         sorted_dict[key] = i_u+self.off_u
            #         i_u += 1
            #     if r > c:
            #         while(r>=next_base_l):
            #             p_base_l+=1
            #             MF_GS_L_Ptr[p_base_l] = i_l
            #             now_base_l = self.color_vertex_num[p_base_l+1]
            #             next_base_l = self.color_vertex_num[p_base_l+2]
            #         self.MF_L_row[i_l] = r
            #         self.MF_L_col[i_l] = c
            #         MF_GS_L_row[i_l] = r-now_base_l
            #         MF_GS_L_col[i_l] = c
            #         sorted_dict[key] = i_l+self.off_l
            #         i_l += 1

            # for i in range(self.N_hexagons):
            #     for j in range(8):
            #         for k in range(8):
            #             id0 = vertex2index[hex[i][j]].item()
            #             id1 = vertex2index[hex[i][k]].item()
            #             key = tuple([id0,id1])
            #             hex_update_offset[i*64+j*8+k] = sorted_dict[key]
            new_dict = {}
            for i in range(self.N_hexagons):
                for j in range(8):
                    for k in range(8):
                        id0 = vertex2index[hex[i][j]].item()
                        id1 = vertex2index[hex[i][k]].item()
                        
                        key = tuple([id0,id1])
                        if id0>id1:
                            if key not in new_dict:
                                while(id0>=next_base_l):
                                    p_base_l+=1
                                    MF_GS_L_Ptr[p_base_l] = i_l
                                    now_base_l = self.color_vertex_num[p_base_l+1]
                                    next_base_l = self.color_vertex_num[p_base_l+2]
                                self.MF_L_row[i_l] = id0
                                self.MF_L_col[i_l] = id1
                                MF_GS_L_row[i_l] = id0-now_base_l
                                MF_GS_L_col[i_l] = id1
                                new_dict[key] = i_l + self.off_l
                                hex_update_offset[i*64+j*8+k] = i_l + self.off_l
                                i_l += 1
                            else:
                                hex_update_offset[i*64+j*8+k]=new_dict[key]
                        if id0 == id1:
                            if key not in new_dict:
                                self.MF_D_row[i_d] = id0
                                self.MF_D_col[i_d] = id1
                                new_dict[key] = id0 + self.off_d
                                hex_update_offset[i*64+j*8+k] = id0 + self.off_d
                                diag_offset[i_d] = id0 + self.off_d
                                i_d += 1
                            else:
                                hex_update_offset[i*64+j*8+k]=new_dict[key]
                        if id0<id1:
                            if key not in new_dict:
                                while(id0>=c_base_u):
                                    p_base_u+=1
                                    MF_GS_U_Ptr[p_base_u] = i_u
                                    r_base_u = self.color_vertex_num[p_base_u]
                                    c_base_u = self.color_vertex_num[p_base_u+1]
                                
                                self.MF_U_row[i_u] = id0
                                self.MF_U_col[i_u] = id1
                                MF_GS_U_row[i_u] = id0-r_base_u
                                MF_GS_U_col[i_u] = id1-c_base_u                          
                                new_dict[key] = i_u + self.off_u
                                hex_update_offset[i*64+j*8+k] = i_u + self.off_u
                                i_u += 1
                            else:
                                hex_update_offset[i*64+j*8+k]=new_dict[key]
            while(p_base_l<self.color_num-1):
                p_base_l+=1
                MF_GS_L_Ptr[p_base_l] = i_l
            while(p_base_u<self.color_num-1):
                p_base_u+=1
                MF_GS_U_Ptr[p_base_u] = i_u
            # for i in range(3):
            #     print(MF_GS_L_Ptr[i].item(),MF_GS_U_Ptr[i].item())
            self.MF_L_value_gpu = wp.zeros(shape=(self.MF_L_nnz),dtype=wp.mat33f,device='cuda:0')
            self.MF_D_value_gpu = wp.zeros(shape=(self.MF_D_nnz),dtype=wp.mat33f,device='cuda:0')
            self.MF_U_value_gpu = wp.zeros(shape=(self.MF_U_nnz),dtype=wp.mat33f,device='cuda:0')
            self.MF_D_row_gpu = wp.from_torch(self.MF_D_row.to('cuda:0'),dtype=wp.int32)
            self.MF_D_col_gpu = wp.from_torch(self.MF_D_col.to('cuda:0'),dtype=wp.int32)
            self.MF_L_row_gpu = wp.from_torch(self.MF_L_row.to('cuda:0'),dtype=wp.int32)
            self.MF_L_col_gpu = wp.from_torch(self.MF_L_col.to('cuda:0'),dtype=wp.int32)
            self.MF_U_row_gpu = wp.from_torch(self.MF_U_row.to('cuda:0'),dtype=wp.int32)
            self.MF_U_col_gpu = wp.from_torch(self.MF_U_col.to('cuda:0'),dtype=wp.int32)
            self.MF_GS_U_col_gpu = wp.from_torch(MF_GS_U_col.to('cuda:0'),dtype=wp.int32)
            self.MF_GS_U_row_gpu = wp.from_torch(MF_GS_U_row.to('cuda:0'),dtype=wp.int32)

            
            self.MF_GS_U_nnz = (MF_GS_U_Ptr[1]-MF_GS_U_Ptr[0]).item()
            self.MF_GS_value_gpu = wp.zeros(shape=(self.MF_GS_U_nnz),dtype=wp.mat33f,device='cuda:0')
            self.MF_GS_U_row_0_gpu = wp.zeros(shape=(self.MF_GS_U_nnz),dtype=wp.int32,device='cuda:0')
            self.MF_GS_U_col_0_gpu = wp.zeros(shape=(self.MF_GS_U_nnz),dtype=wp.int32,device='cuda:0')
            wp.copy(self.MF_GS_U_row_0_gpu,self.MF_D_row_gpu,count=self.MF_GS_U_nnz)
            wp.copy(self.MF_GS_U_col_0_gpu,self.MF_D_col_gpu,count=self.MF_GS_U_nnz)

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

        self.IM_gpu = wp.from_torch(IM.to('cuda:0'),dtype=wp.mat33f)
        self.hexagons_gpu = wp.from_torch(hexagons.to('cuda:0'))
        self.shapeFuncGrad_gpu = wp.from_torch(shapeFuncGrad.to('cuda:0'))
        self.pin_pos_gpu = wp.from_torch(pin_pos.to('cuda:0'),dtype=wp.vec3)
        self.pin_list_gpu = wp.from_torch(pin_list.to('cuda:0'))
        self.dev_vertex2index = wp.from_torch(vertex2index.to('cuda:0'),dtype=wp.int32)
        self.dev_index2vertex = wp.from_torch(index2vertex.to('cuda:0'),dtype=wp.int32)

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
        self.D = warp.sparse.bsr_zeros(self.N_verts,self.N_verts,wp.mat33f,device='cuda:0')

        self.D_GS = warp.sparse.bsr_zeros(self.color_vertex_num[1]-self.color_vertex_num[0],self.color_vertex_num[2]-self.color_vertex_num[1],wp.mat33f,device='cuda:0') 
        
        self.X = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.color_x_gpu_0 = wp.zeros((self.color_vertex_num[1]-self.color_vertex_num[0]),dtype=wp.vec3,device='cuda:0')
        self.color_x_gpu_1 = wp.zeros((self.color_vertex_num[2]-self.color_vertex_num[1]),dtype=wp.vec3,device='cuda:0')
        self.MF_value_gpu = wp.zeros(shape=(self.MF_nnz),dtype=wp.mat33f,device='cuda:0')
        self.dev_temp_X = wp.zeros(shape=(self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.squared_sum = wp.zeros(shape=(1),dtype=wp.float32,device='cuda:0')

    def compute_A_and_B(self):
        start = time.time()
        wp.launch(kernel=compute_elastic_hessian,dim=self.N_hexagons*64,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.MF_value_gpu,self.hex_update_offset_gpu])
        wp.synchronize()
        end = time.time()
        print('Time : ',end-start)


    # def gradientDescent(self,iterations = 100,lr = 1e-3):
    #     for iter in range(iterations):
    #         self.grad_gpu.zero_()
    #         self.energy.zero_()
    #         wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy])
    #         wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_gpu,self.m_gpu,self.g_gpu,self.energy])
    #         wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.grad_gpu])
    #         wp.launch(kernel=compute_partial_gravity_energy_X,dim=self.N_verts,inputs=[self.m_gpu,self.g_gpu,self.grad_gpu])
    #         alpha = lr
    #         for i in range(20):
    #             wp.launch(kernel=minues_grad,dim=self.N_verts,inputs=[self.x_gpu,self.x_lineSearch,self.grad_gpu,alpha])
    #             wp.launch(kernel=pin,dim=self.N_pin,inputs=[self.x_lineSearch,self.pin_pos_gpu,self.pin_list_gpu])
    #             self.energy_lineSearch.zero_()
    #             wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_lineSearch,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy_lineSearch])
    #             wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_lineSearch,self.m_gpu,self.g_gpu,self.energy_lineSearch])
    #             wp.synchronize()
    #             if self.energy_lineSearch.numpy()[0] < self.energy.numpy()[0]:
    #                 wp.copy(self.x_gpu,self.x_lineSearch)
    #                 break
    #             else:
    #                 alpha = alpha*0.2
    #         wp.synchronize()
    #         if iter%(iterations/100) == 0:
    #             print('iter :',iter,' Energy : ',self.energy.numpy()[0])

    def PerformJacobi(self,iterations = 10):
        #self.X.zero_()
        wp.copy(self.MF_L_value_gpu,self.MF_value_gpu,dest_offset=0,src_offset=self.off_l,count=self.MF_L_nnz)
        wp.copy(self.MF_D_value_gpu,self.MF_value_gpu,dest_offset=0,src_offset=self.off_d,count=self.MF_D_nnz)
        wp.copy(self.MF_U_value_gpu,self.MF_value_gpu,dest_offset=0,src_offset=self.off_u,count=self.MF_U_nnz)
        warp.sparse.bsr_set_from_triplets(self.L,self.MF_L_row_gpu,self.MF_L_col_gpu,self.MF_L_value_gpu)
        warp.sparse.bsr_set_from_triplets(self.U,self.MF_U_row_gpu,self.MF_U_col_gpu,self.MF_U_value_gpu)
        warp.sparse.bsr_set_from_triplets(self.D,self.MF_D_row_gpu,self.MF_D_col_gpu,self.MF_D_value_gpu)
        for iter in range(iterations):
            wp.copy(self.dev_temp_X,self.grad_gpu)
            # warp.sparse.bsr_mv(self.L,self.X,self.dev_temp_X,alpha=-1.0,beta=1.0)
            # warp.sparse.bsr_mv(self.U,self.X,self.dev_temp_X,alpha=-1.0,beta=1.0)
            wp.launch(kernel=jacobi_iteration,dim=self.N_verts,inputs=[self.X,self.MF_D_value_gpu,self.dev_temp_X])


    def PerformGaussSeidel(self,iterations = 10):
        wp.copy(self.MF_U_value_gpu,self.MF_value_gpu,dest_offset=0,src_offset=self.off_u,count=self.MF_U_nnz)
        wp.copy(self.MF_D_value_gpu,self.MF_value_gpu,dest_offset=0,src_offset=self.off_d,count=self.MF_D_nnz)
        wp.copy(self.MF_L_value_gpu,self.MF_value_gpu,dest_offset=0,src_offset=self.off_l,count=self.MF_L_nnz)
        warp.sparse.bsr_set_from_triplets(self.L,self.MF_L_row_gpu,self.MF_L_col_gpu,self.MF_L_value_gpu)
        warp.sparse.bsr_set_from_triplets(self.U,self.MF_U_row_gpu,self.MF_U_col_gpu,self.MF_U_value_gpu)

        wp.copy(self.MF_GS_value_gpu,self.MF_U_value_gpu,dest_offset=0,src_offset=0,count=self.MF_GS_U_nnz)
        warp.sparse.bsr_set_from_triplets(self.D_GS,self.MF_GS_U_row_0_gpu,self.MF_GS_U_col_0_gpu,self.MF_GS_value_gpu)

        for iter in range(iterations):
            wp.copy(self.dev_temp_X,self.grad_gpu)
            warp.sparse.bsr_mv(self.L,self.X,self.dev_temp_X,alpha=-1.0,beta=1.0)
            self.X.zero_()
            for color in range(self.color_num-1,-1,-1):
                base = self.color_vertex_num[color]
                wp.launch(kernel=Colored_GS_MF_Kernel,dim=self.color_vertex_num[color+1]-base,inputs=[self.X,self.MF_D_value_gpu,self.dev_temp_X,base,self.color_vertex_num[color+1]-base])
                if color:
                    wp.copy(self.color_x_gpu_1,self.X,dest_offset=0,src_offset=self.color_vertex_num[color],count=self.color_vertex_num[color+1]-self.color_vertex_num[color])
                    wp.copy(self.color_x_gpu_0,self.dev_temp_X,0,self.color_vertex_num[color-1],self.color_vertex_num[color]-self.color_vertex_num[color-1])
                    warp.sparse.bsr_mv(self.D_GS,self.color_x_gpu_1,self.color_x_gpu_0,alpha=-1.0,beta=1.0)
                    wp.copy(self.dev_temp_X,self.color_x_gpu_0,self.color_vertex_num[color-1],0,self.color_vertex_num[color]-self.color_vertex_num[color-1])
            wp.copy(self.dev_temp_X,self.grad_gpu)
            warp.sparse.bsr_mv(self.U,self.X,self.dev_temp_X,alpha=-1.0,beta=1.0)
            self.X.zero_()
            for color in range(self.color_num):
                base = self.color_vertex_num[color]
                wp.launch(kernel=Colored_GS_MF_Kernel,dim=self.color_vertex_num[color+1]-base,inputs=[self.X,self.MF_D_value_gpu,self.dev_temp_X,base,self.color_vertex_num[color+1]-base])
                if color < self.color_num-1:
                    wp.copy(self.color_x_gpu_0,self.X,0,0,count=self.color_vertex_num[color+1])
                    wp.copy(self.color_x_gpu_1,self.dev_temp_X,0,self.color_vertex_num[color+1],count=self.color_vertex_num[color+1])
                    




    def Newton(self,iterations = 100):
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
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.dev_vertex2index,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.grad_gpu])
            wp.launch(kernel=compute_partial_gravity_energy_X,dim=self.N_verts,inputs=[self.m_gpu,self.g_gpu,self.grad_gpu])
            wp.launch(kernel=compute_elastic_hessian,dim=self.N_hexagons*64,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.MF_value_gpu,self.hex_update_offset_gpu])
            wp.launch(kernel=spd_matrix33f,dim=self.MF_nnz,inputs=[self.MF_value_gpu])
            #self.x.zero_()
            if self.store_LDU == 0:
                wp.launch(kernel=jacobi_iteration_offset,dim=self.N_verts,inputs=[self.X,self.MF_value_gpu,self.diag_offset_gpu,self.grad_gpu])
            else:
                #if self.Jacobi:
                self.PerformJacobi(1)
                # wp.copy(self.dev_temp_X,self.grad_gpu)
                # self.squared_sum.zero_()
                # wp.launch(kernel=square_sum,dim=self.N_verts,inputs=[self.dev_temp_X,self.squared_sum])
                # print('before solve squared_sum : ',self.squared_sum.numpy()[0])
                # warp.sparse.bsr_mv(self.L,self.X,self.dev_temp_X,alpha=-1.0,beta=1.0)
                # warp.sparse.bsr_mv(self.U,self.X,self.dev_temp_X,alpha=-1.0,beta=1.0)
                # warp.sparse.bsr_mv(self.D,self.X,self.dev_temp_X,alpha=-1.0,beta=1.0)
                # self.squared_sum.zero_()
                # wp.launch(kernel=square_sum,dim=self.N_verts,inputs=[self.dev_temp_X,self.squared_sum])
                # print('after solve  squared_sum : ',self.squared_sum.numpy()[0])
            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.x_gpu,self.X,self.dev_index2vertex])
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
import pyvista as pv
import warp as wp
from sparse import *
import torch            #warp只能在gpu上运行，所以这里用torch来做一些cpu上的操作再运到warp上
import numpy as np
from hexagon import *
from cublas import *
from cpu_function import *
import time
import sys
import operator
import matplotlib.pyplot as plt

class Object: 

    def __init__(self,mesh_path,dx,pinList):
        # 层数
        self.layer = 3
        

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

        #找出包围盒左下角的点，用于计算相对位置
        self.box_min = torch.tensor([1000.0,1000.0,1000.0],dtype=torch.float32)
        for i in range(self.N_verts):
            for j in range(3):
                if x[i][j]<self.box_min[j]:
                    self.box_min[j] = x[i][j]
        
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
        self.Ut = [bsr_zeros(self.dims[i+1],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer-1)]
        self.Us = [bsr_zeros(self.dims[i],self.dims[i+1],wp.mat33f,device='cuda:0') for i in range(self.layer-1)]
        
        for layer in range(self.layer-1):
            rowInd = []
            colInd = []
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
                                        value.append(1.0)
                                    #最中间那个点，插值系数是0.125，给8个顶点插值
                                    elif i%2+j%2 +k%2 == 3:
                                        for l in range(8):
                                            coarse_id = self.hexs[layer+1][HEX][l]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            value.append(0.125)
                                    # 12个边的中点，插值系数是0.5，给2个顶点插值
                                    elif i%2+j%2+k%2 == 1:
                                        if i == 1:
                                            coarse_id = self.hexs[layer+1][HEX][4*0+2*(j//2)+(k//2)]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            value.append(0.5)
                                            coarse_id = self.hexs[layer+1][HEX][4*1+2*(j//2)+(k//2)]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            value.append(0.5)
                                        elif j == 1:
                                            coarse_id = self.hexs[layer+1][HEX][4*(i//2)+2*0+(k//2)]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            value.append(0.5)
                                            coarse_id = self.hexs[layer+1][HEX][4*(i//2)+2*1+(k//2)]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            value.append(0.5)
                                        elif k == 1:
                                            coarse_id = self.hexs[layer+1][HEX][4*(i//2)+2*(j//2)+0]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            value.append(0.5)
                                            coarse_id = self.hexs[layer+1][HEX][4*(i//2)+2*(j//2)+1]
                                            rowInd.append(vertex2index[layer+1][coarse_id])
                                            colInd.append(vertex2index[layer][fine_id])
                                            value.append(0.5)
                                    #6个面的中心，给4个顶点插值，插值系数是0.25       
                                    elif i%2+j%2+k%2 == 2:
                                        if i%2==0:
                                            for ii in range(2):
                                                for jj in range(2):
                                                    coarse_id = self.hexs[layer+1][HEX][4*(i//2)+2*ii+jj]
                                                    rowInd.append(vertex2index[layer+1][coarse_id])
                                                    colInd.append(vertex2index[layer][fine_id])
                                                    value.append(0.25)
                                        elif j%2==0:
                                            for ii in range(2):
                                                for jj in range(2):
                                                    coarse_id = self.hexs[layer+1][HEX][4*ii+2*(j//2)+jj]
                                                    rowInd.append(vertex2index[layer+1][coarse_id])
                                                    colInd.append(vertex2index[layer][fine_id])
                                                    value.append(0.25)
                                        elif k%2==0:
                                            for ii in range(2):
                                                for jj in range(2):
                                                    coarse_id = self.hexs[layer+1][HEX][4*ii+2*jj+(k//2)]
                                                    rowInd.append(vertex2index[layer+1][coarse_id])
                                                    colInd.append(vertex2index[layer][fine_id])
                                                    value.append(0.25)
            Us_rowInd = wp.from_torch(torch.tensor(rowInd,dtype=torch.int32).to('cuda:0'))
            Us_colInd = wp.from_torch(torch.tensor(colInd,dtype=torch.int32).to('cuda:0'))
            Us_value = torch.zeros((len(value),3,3),dtype=torch.float32)
            for i in range(len(value)):
                Us_value[i] = torch.eye(3,dtype=torch.float32)*value[i]
            Us_values = wp.from_torch(Us_value.to('cuda:0'),dtype=wp.mat33f) 
            bsr_set_from_triplets(self.Ut[layer],Us_rowInd,Us_colInd,Us_values)
        
        #Us和Ut这样写是为了和tiantian liu的论文对齐
        for layer in range(self.layer-1):
            self.Us[layer] = bsr_transposed(self.Ut[layer])
        print('U matrix build done')    
                                   
                                
        
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
                    # if r<10:
                    #      print(r,c,c_base_u)
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
                            hex_update_offset[i*64+j*8+k] = sorted_dict[key]

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
                self.hex_update_offset_gpu = wp.from_torch(hex_update_offset.to('cuda:0'),dtype=wp.int32)

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
                        i_l += 1
                        i_f += 1
                while(p_base_l<self.color_num[layer]-1):
                    p_base_l+=1
                    self.UtAUs_GS_L_Ptr[layer-1][p_base_l] = i_l
                while(p_base_u<self.color_num[layer]-1):
                    p_base_u+=1
                    self.UtAUs_GS_U_Ptr[layer-1][p_base_u] = i_u        
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
        self.dev_vertex2index = wp.from_torch(vertex2index[0].to('cuda:0'),dtype=wp.int32)
        self.dev_index2vertex = wp.from_torch(index2vertex[0].to('cuda:0'),dtype=wp.int32)

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
        self.A = [bsr_zeros(self.dims[i],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer)]
        self.L = [bsr_zeros(self.dims[i],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer)]
        self.U = [bsr_zeros(self.dims[i],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer)]
        self.D = [bsr_zeros(self.dims[i],self.dims[i],wp.mat33f,device='cuda:0') for i in range(self.layer)]

        self.MF_GS_U = [bsr_zeros(self.color_vertex_num[0][i+1]-self.color_vertex_num[0][i],self.color_vertex_num[0][self.color_num[0]]-self.color_vertex_num[0][i+1],wp.mat33f,device='cuda:0') for i in range(self.color_num[0]-1)]
        self.MF_GS_L = [bsr_zeros(self.color_vertex_num[0][i+2]-self.color_vertex_num[0][i+1],self.color_vertex_num[0][i+1],wp.mat33f,device='cuda:0') for i in range(self.color_num[0]-1)]
        
        self.X = wp.zeros((self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.dev_R = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_B = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_B_fixed = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_delta_x = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]
        self.dev_x_solved = [wp.zeros((self.dims[i]),dtype=wp.vec3,device='cuda:0') for i in range(self.layer)]

        self.MF_value_gpu = wp.zeros(shape=(self.MF_nnz),dtype=wp.mat33f,device='cuda:0')
        self.dev_temp_X = wp.zeros(shape=(self.N_verts),dtype=wp.vec3,device='cuda:0')
        self.squared_sum = wp.zeros(shape=(1),dtype=wp.float32,device='cuda:0')

        self.UtAUs_D_row_gpu = [wp.from_torch(self.UtAUs_D_row[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_D_col_gpu = [wp.from_torch(self.UtAUs_D_col[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_L_row_gpu = [wp.from_torch(self.UtAUs_L_row[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_L_col_gpu = [wp.from_torch(self.UtAUs_L_col[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]    
        self.UtAUs_U_row_gpu = [wp.from_torch(self.UtAUs_U_row[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_U_col_gpu = [wp.from_torch(self.UtAUs_U_col[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_GS_U_row_gpu = [wp.from_torch(self.UtAUs_GS_U_row[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_GS_U_col_gpu = [wp.from_torch(self.UtAUs_GS_U_col[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_GS_L_row_gpu = [wp.from_torch(self.UtAUs_GS_L_row[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_GS_L_col_gpu = [wp.from_torch(self.UtAUs_GS_L_col[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]

        self.UtAUs_GS_U = []
        self.UtAUs_GS_L = []
        for layer in range(self.layer-1):
            self.UtAUs_GS_U.append([bsr_zeros(self.color_vertex_num[layer+1][i+1]-self.color_vertex_num[layer+1][i],self.color_vertex_num[layer+1][self.color_num[layer+1]]-self.color_vertex_num[layer+1][i+1],wp.mat33f,device='cuda:0') for i in range(self.color_num[layer+1]-1)])
            self.UtAUs_GS_L.append([bsr_zeros(self.color_vertex_num[layer+1][i+2]-self.color_vertex_num[layer+1][i+1],self.color_vertex_num[layer+1][i+1],wp.mat33f,device='cuda:0') for i in range(self.color_num[layer+1]-1)])
        
        self.UtAUs_block_offset_gpu = [wp.from_torch(self.UtAUs_block_offset[i].to('cuda:0'),dtype=wp.int32) for i in range(self.layer-1)]
        self.UtAUs_values = [wp.zeros(shape=(self.UtAUs_nnz[i]),dtype=wp.mat33f,device='cuda:0') for i in range(self.layer-1)]
            
        self.plot_x = []
        self.plot_y = []

        print('init done')
        

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



    def downSample(self,layer = 0):
        wp.copy(self.dev_R[layer],self.dev_B_fixed[layer])
        wp.copy(self.dev_x_solved[layer],self.dev_delta_x[layer])
        bsr_mv(self.L[layer],self.dev_delta_x[layer],self.dev_R[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.U[layer],self.dev_delta_x[layer],self.dev_R[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.D[layer],self.dev_delta_x[layer],self.dev_R[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.Ut[layer],self.dev_R[layer],self.dev_B_fixed[layer+1],alpha=1.0,beta=0.0)
    
    def upSample(self,layer = 1):
        #print(self.dev_delta_x[layer])
        wp.launch(kernel=addVec3,dim=self.dims[layer],inputs=[self.dev_x_solved[layer],self.dev_delta_x[layer]])
        bsr_mv(self.Us[layer-1],self.dev_x_solved[layer],self.dev_x_solved[layer-1],alpha=1.0,beta=1.0)
        bsr_mv(self.L[layer-1],self.dev_x_solved[layer-1],self.dev_B_fixed[layer-1],alpha=-1.0,beta=1.0)
        bsr_mv(self.D[layer-1],self.dev_x_solved[layer-1],self.dev_B_fixed[layer-1],alpha=-1.0,beta=1.0)
        bsr_mv(self.U[layer-1],self.dev_x_solved[layer-1],self.dev_B_fixed[layer-1],alpha=-1.0,beta=1.0)
        
    def finish(self):
        wp.launch(kernel=addVec3,dim=self.dims[0],inputs=[self.dev_x_solved[0],self.dev_delta_x[0]])
        for i in range(1,self.layer):
            self.dev_x_solved[i].zero_()


         
    def showError(self,layer):
        print('layer : ',layer)
        wp.copy(self.dev_B[layer],self.dev_B_fixed[layer])
        self.squared_sum.zero_()
        wp.launch(kernel=square_sum,dim=self.dims[layer],inputs=[self.dev_B[layer],self.squared_sum])
        print('before solve squared_sum : ',self.squared_sum.numpy()[0])
        bsr_mv(self.L[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.U[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
        bsr_mv(self.D[layer],self.dev_delta_x[layer],self.dev_B[layer],alpha=-1.0,beta=1.0)
        self.squared_sum.zero_()
        wp.launch(kernel=square_sum,dim=self.dims[layer],inputs=[self.dev_B[layer],self.squared_sum])
        print('after solve  squared_sum : ',self.squared_sum.numpy()[0])
        # if layer == 2:
        #     print(self.dev_delta_x[layer])

    def Newton(self,iterations = 1000):
        for step in range(iterations):
            if step%(iterations/10) == 0:
                wp.synchronize()
                print('Step : ',step)
                self.energy.zero_()
                wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy])
                wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_gpu,self.m_gpu,self.g_gpu,self.energy])
                print('Energy : ',self.energy.numpy()[0])
                self.plot_x.append(step)
                self.plot_y.append(self.energy.numpy()[0])
            self.grad_gpu.zero_()
            self.MF_value_gpu.zero_()
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.dev_vertex2index,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.grad_gpu])
            wp.launch(kernel=compute_partial_gravity_energy_X,dim=self.N_verts,inputs=[self.m_gpu,self.g_gpu,self.grad_gpu])
            wp.launch(kernel=compute_elastic_hessian,dim=self.N_hexagons*64,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.MF_value_gpu,self.hex_update_offset_gpu])
            wp.launch(kernel=spd_matrix33f,dim=self.MF_nnz,inputs=[self.MF_value_gpu])
            bsr_set_from_triplets(self.L[0],self.MF_L_row_gpu,self.MF_L_col_gpu,self.MF_value_gpu,value_offset=self.off_l)
            bsr_set_from_triplets(self.U[0],self.MF_U_row_gpu,self.MF_U_col_gpu,self.MF_value_gpu,value_offset=self.off_u)
            bsr_set_from_triplets(self.D[0],self.MF_D_row_gpu,self.MF_D_col_gpu,self.MF_value_gpu,value_offset=self.off_d)

            wp.copy(self.dev_B_fixed[0],self.grad_gpu)
            #self.PerformJacobi(layer=0,iterations=1)
            self.PerformGaussSeidel(layer=0,iterations=3)

            #self.showError(0)
            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.x_gpu,self.dev_delta_x[0],self.dev_index2vertex])
            wp.launch(kernel=pin,dim=self.N_pin,inputs=[self.x_gpu,self.pin_pos_gpu,self.pin_list_gpu])

    def NewtonMultigrid(self,iterations = 100):
        for step in range(iterations+1):
            if step%(iterations/10) == 0:
                wp.synchronize()
                print('Step : ',step)
                self.energy.zero_()
                wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy])
                wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_gpu,self.m_gpu,self.g_gpu,self.energy])
                print('Energy : ',self.energy.numpy()[0])
                self.plot_x.append(step)
                self.plot_y.append(self.energy.numpy()[0])
            self.grad_gpu.zero_()
            self.MF_value_gpu.zero_()
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu,self.hexagons_gpu,self.dev_vertex2index,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.grad_gpu])
            wp.launch(kernel=compute_partial_gravity_energy_X,dim=self.N_verts,inputs=[self.m_gpu,self.g_gpu,self.grad_gpu])
            wp.launch(kernel=compute_elastic_hessian,dim=self.N_hexagons*64,inputs=[self.x_gpu,self.hexagons_gpu,self.shapeFuncGrad_gpu,self.det_pX_peps_gpu,self.inverse_pX_peps_gpu,self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.MF_value_gpu,self.hex_update_offset_gpu])
            wp.launch(kernel=spd_matrix33f,dim=self.MF_nnz,inputs=[self.MF_value_gpu])
            for i in range(self.layer):
                if i == 0:
                    bsr_set_from_triplets(self.A[i],self.MF_row_gpu,self.MF_col_gpu,self.MF_value_gpu)
                    bsr_set_from_triplets(self.L[i],self.MF_L_row_gpu,self.MF_L_col_gpu,self.MF_value_gpu,value_offset=self.off_l)
                    bsr_set_from_triplets(self.U[i],self.MF_U_row_gpu,self.MF_U_col_gpu,self.MF_value_gpu,value_offset=self.off_u)
                    bsr_set_from_triplets(self.D[i],self.MF_D_row_gpu,self.MF_D_col_gpu,self.MF_value_gpu,value_offset=self.off_d)
                else:
                    self.A[i] = bsr_mm(self.Ut[i-1],bsr_mm(self.A[i-1],self.Us[i-1]))
                    wp.launch(kernel=spd_matrix33f,dim=self.UtAUs_nnz[i-1],inputs=[self.A[i].values])
                    wp.launch(kernel = block_values_reorder,dim = self.UtAUs_nnz[i-1],inputs=[self.A[i].values,self.UtAUs_values[i-1],self.UtAUs_block_offset_gpu[i-1]])                    
                    bsr_set_from_triplets(self.L[i],self.UtAUs_L_row_gpu[i-1],self.UtAUs_L_col_gpu[i-1],self.UtAUs_values[i-1],value_offset=self.UtAUs_off_l[i-1])
                    bsr_set_from_triplets(self.U[i],self.UtAUs_U_row_gpu[i-1],self.UtAUs_U_col_gpu[i-1],self.UtAUs_values[i-1],value_offset=self.UtAUs_off_u[i-1])
                    bsr_set_from_triplets(self.D[i],self.UtAUs_D_row_gpu[i-1],self.UtAUs_D_col_gpu[i-1],self.UtAUs_values[i-1],value_offset=self.UtAUs_off_d[i-1])

            wp.copy(self.dev_B_fixed[0],self.grad_gpu)
            #V_cycle

            #self.PerformJacobi(layer=0,iterations=1)
            self.PerformGaussSeidel(layer=0,iterations=1)
            self.showError(layer=0)

            self.downSample(layer=0)

            self.PerformGaussSeidel(layer=1,iterations=1)
            self.showError(layer=1)

            self.downSample(layer=1)
            self.PerformGaussSeidel(layer=2,iterations=5)
            self.showError(layer=2)
            self.upSample(layer=2)
            self.PerformGaussSeidel(layer=1,iterations=1)
            self.showError(layer=1)

            self.upSample(layer=1)

            self.PerformGaussSeidel(layer=0,iterations=1)
            self.showError(layer=0)

            self.finish()


            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.x_gpu,self.dev_x_solved[0],self.dev_index2vertex])
            wp.launch(kernel=pin,dim=self.N_pin,inputs=[self.x_gpu,self.pin_pos_gpu,self.pin_list_gpu])

    def show_layer(self,layer=0):
        index = np.array([0,4,6,2,1,5,7,3])
        cells = self.hexs[layer].numpy()[:,index]
        hex_num = np.zeros((cells.shape[0],1),dtype=np.int32)
        hex_num.fill(8)
        cells = np.concatenate((hex_num,cells),axis=1)
        cell_type = np.zeros(cells.shape[0],dtype=np.int8)
        cell_type.fill(pv.CellType.HEXAHEDRON)
        grid = pv.UnstructuredGrid(cells, cell_type, self.x[layer].numpy())
        grid.plot(show_edges=True)

    def show(self):
        # 创建画布和子图
        fig, ax = plt.subplots()

        # 绘制线图
        ax.plot(self.plot_x, self.plot_y, linestyle='-', color='blue', label='Line')

        # 添加标题和标签
        ax.set_title('Plot')
        ax.set_xlabel('iterations')
        ax.set_ylabel('energy')

        # 添加图例
        ax.legend()

        # 显示图形
        plt.show()

        wp.copy(self.x_cpu,self.x_gpu)
        plot_verts = self.x[0].numpy()
        self.voxels.points = plot_verts
        pl = pv.Plotter()  
        pl.add_mesh(self.voxels, color=True, show_edges=True)
        # Set camera start position
        pl.camera_position = 'xy'
        pl.show()  
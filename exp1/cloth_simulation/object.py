import pyvista as pv
import warp as wp
from sparse import *
import torch            #warp只能在gpu上运行，所以这里用torch来做一些cpu上的操作再运到warp上
import numpy as np
from triangle import *
from cublas import *
from cpu_function import *
import time
import sys
import operator
import matplotlib.pyplot as plt


from cuda import cudart
from OpenGL.GL import *

EPSILON = 1e-7
class Cloth: 

    def __init__(self):
        # 层数
        self.layer = 1
        self.control_mag = 10.0
        self.spd_value = 1e-3
        self.dt = 0.0033
        self.inv_dt = 1.0/self.dt
        self.damping = 0.9995
        self.k = 1000.0

        self.res_x = 64
        self.res_y = 64
        #生成布料的顶点，面，边数据
        #顶点数据
        self.x_cpu = torch.zeros(((self.res_x+1)*(self.res_y+1),3),dtype=torch.float32,requires_grad=False)
        for i in range(self.x_cpu.shape[0]):
            x = i%(self.res_x+1)
            y = i//(self.res_x+1)
            self.x_cpu[i][0] = x*(1.0/self.res_x)
            self.x_cpu[i][1] = y*(1.0/self.res_y)
            self.x_cpu[i][2] = 0.0
        self.x_gpu = wp.from_torch(self.x_cpu.to('cuda:0'),dtype=wp.vec3f)
        #面数据
        self.face = torch.zeros((self.res_x*self.res_y*2,3),dtype=torch.int32,requires_grad=False)
        for i in range(self.res_y):
            for j in range(self.res_x):
                idx = 2*(i*self.res_x+j)
                self.face[idx][0] = i*(self.res_x+1)+j
                self.face[idx][1] = i*(self.res_x+1)+j+1
                self.face[idx][2] = (i+1)*(self.res_x+1)+j
                idx += 1
                self.face[idx][0] = i*(self.res_x+1)+j+1
                self.face[idx][1] = (i+1)*(self.res_x+1)+j+1
                self.face[idx][2] = (i+1)*(self.res_x+1)+j
        
        #边数据
        self.edge = torch.zeros((self.res_x*(self.res_y+1)+self.res_y*(self.res_x+1)+self.res_x*self.res_y,2),dtype=torch.int32,requires_grad=False)
        for i in range(self.res_y+1):
            for j in range(self.res_x):
                idx = i*self.res_x+j
                self.edge[idx][0] = i*(self.res_x+1)+j
                self.edge[idx][1] = i*(self.res_x+1)+j+1
        for i in range(self.res_x+1):
            for j in range(self.res_y):
                idx = (self.res_y+1)*self.res_x+i*self.res_y+j
                self.edge[idx][0] = j*(self.res_x+1)+i
                self.edge[idx][1] = (j+1)*(self.res_x+1)+i
        for i in range(self.res_y):
            for j in range(self.res_x):
                idx = (self.res_y+1)*self.res_x+(self.res_x+1)*self.res_y+i*self.res_x+j
                self.edge[idx][0] = i*(self.res_x+1)+j
                self.edge[idx][1] = (i+1)*(self.res_x+1)+j+1
        self.edge_gpu = wp.from_torch(self.edge.to('cuda:0'),dtype=wp.vec2i)

        self.N_verts = int(self.x_cpu.shape[0])
        self.N_face = int(self.face.shape[0])
        self.N_edge = int(self.edge.shape[0])
        self.surface = self.face.numpy()
        self.face_gpu = wp.from_torch(self.face.to('cuda:0'),dtype = wp.vec3i)
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
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * self.surface.size,self.surface , GL_STATIC_DRAW)
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

        #拾取相关  估计光线最多穿过128个面
        self.select_num = warp.zeros((1),dtype=wp.int32,device='cuda:0')
        self.select_faces = warp.zeros((128),dtype=wp.int32,device='cuda:0')
        self.select_distances = warp.zeros((128),dtype=wp.float32,device='cuda:0')
        self.select_vertex = -1
        self.x_numpy = np.zeros((self.N_verts,3),dtype=np.float32)
        self.target = glm.vec3(0.0,0.0,0.0)

        #simulation component
        #弹簧原长度
        self.rest_length = wp.zeros((self.N_edge),dtype=wp.float32,device='cuda:0')
        wp.launch(kernel=compute_rest_length,dim=self.N_edge,inputs=[self.x_gpu,self.edge_gpu,self.rest_length])
        #顶点质量
        self.m_gpu = wp.ones((self.N_verts),dtype=wp.float32,device='cuda:0')

        self.MF_nnz = self.N_verts+self.N_edge*2
        self.MF_value_gpu = wp.zeros((self.MF_nnz),dtype=wp.mat33f,device='cuda:0')
        self.MF_col = torch.zeros((self.MF_nnz),dtype=torch.int32,requires_grad=False)
        self.MF_row = torch.zeros((self.MF_nnz),dtype=torch.int32,requires_grad=False)
        offset = torch.zeros((4*self.N_edge),dtype=torch.int32,requires_grad=False)
        self.d_offset = torch.zeros((self.N_verts),dtype=torch.int32,requires_grad=False)

        edge_hash = {}
        for i in range(self.N_edge):
            id0 = self.edge[i][0].item()
            id1 = self.edge[i][1].item()
            key = tuple([id0,id1])
            edge_hash[key] = i
            key = tuple([id1,id0])
            edge_hash[key] = i+self.N_edge
        
        for i in range(self.N_verts):
            key = tuple([i,i])
            edge_hash[key] = self.N_edge*2+i
        
        sorted_dict = dict(sorted(edge_hash.items(), key=operator.itemgetter(0)))

        index = 0

        for key in sorted_dict.keys():
            r = key[0]
            c = key[1]
            self.MF_row[index] = r
            self.MF_col[index] = c
            sorted_dict[key] = index
            if r==c:
                self.d_offset[r] = index
            index += 1
        
        for i in range(self.N_edge):
            id0 = self.edge[i][0].item()
            id1 = self.edge[i][1].item()
            offset[4*i] = sorted_dict[tuple([id0,id0])]
            offset[4*i+1] = sorted_dict[tuple([id0,id1])]
            offset[4*i+2] = sorted_dict[tuple([id1,id0])]
            offset[4*i+3] = sorted_dict[tuple([id1,id1])]
        
        self.offset_gpu = wp.from_torch(offset.to('cuda:0'),dtype=wp.int32)
        self.d_offset_gpu = wp.from_torch(self.d_offset.to('cuda:0'),dtype=wp.int32)
        self.MF_row_gpu = wp.from_torch(self.MF_row.to('cuda:0'),dtype=wp.int32)
        self.MF_col_gpu = wp.from_torch(self.MF_col.to('cuda:0'),dtype=wp.int32)
        self.dev_x = wp.zeros((self.N_verts),dtype=wp.vec3f,device='cuda:0')
        wp.copy(self.dev_x,self.x_gpu)

        self.dev_v = wp.zeros_like(self.dev_x,device='cuda:0')
        self.dev_old_x = wp.zeros_like(self.dev_x,device='cuda:0')
        self.dev_inertia_x = wp.zeros_like(self.dev_x,device='cuda:0')
        self.dev_B = wp.zeros_like(self.dev_x,device='cuda:0')
        self.dev_P = wp.zeros_like(self.dev_x,device='cuda:0')
        self.dev_AP = wp.zeros_like(self.dev_x,device='cuda:0')
        self.dev_delta_x = wp.zeros_like(self.dev_x,device='cuda:0')
        self.dev_x_solved = wp.zeros_like(self.dev_x,device='cuda:0')
        self.grad_gpu = wp.zeros_like(self.dev_x,device='cuda:0')
        self.dev_B_fixed = wp.zeros_like(self.dev_x,device='cuda:0')

        self.squared_sum = wp.zeros((1),dtype=wp.float32,device='cuda:0')
        self.dot_sum = wp.zeros((1),dtype=wp.float32,device='cuda:0')
        self.norm_max = wp.zeros((1),dtype=wp.float32,device='cuda:0')
        self.g = -9.8

        self.A = bsr_zeros(self.N_verts,self.N_verts,wp.mat33f,device='cuda:0')
        


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

    def PerformConjugateGradient(self,iterations = 10,tol = 1e-5):
        self.dev_delta_x.zero_()
        r0 = 0.0
        r1 = 0.0
        dot = 0.0
        alpha = 0.0
        beta = 0.0
        neg_alpha = 0.0
        wp.copy(self.dev_B,self.dev_B_fixed)
        self.squared_sum.zero_()
        wp.launch(kernel = square_sum,dim=self.N_verts,inputs=[self.dev_B,self.squared_sum])
        r1 = self.squared_sum.numpy()[0]
        
        if r1 < EPSILON:
            return
        r = r1
        k = 1
        while(r1>tol*r and k<=iterations):
            if k > 1:
                beta = r1/r0
                wp.launch(scal,dim=self.N_verts,inputs=[self.dev_P,beta])
                wp.launch(axpy,dim=self.N_verts,inputs=[self.dev_P,self.dev_B,1.0])
            else :
                wp.copy(self.dev_P,self.dev_B_fixed)
            self.dev_AP.zero_()
            bsr_mv(self.A,self.dev_P,self.dev_AP,alpha=1.0,beta=1.0)
    
            self.dot_sum.zero_()
            wp.launch(kernel = cublasSdot,dim=self.N_verts,inputs=[self.dev_P,self.dev_AP,self.dot_sum])
            dot = self.dot_sum.numpy()[0]
            if dot<1e-10:
                break
            alpha = r1/dot
            neg_alpha = -alpha
            wp.launch(axpy,dim=self.N_verts,inputs=[self.dev_delta_x,self.dev_P,alpha])
            wp.launch(axpy,dim=self.N_verts,inputs=[self.dev_B,self.dev_AP,neg_alpha])
            r0 = r1
            self.squared_sum.zero_()
            wp.launch(kernel = square_sum,dim=self.N_verts,inputs=[self.dev_B,self.squared_sum])
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

    def showError(self):
        wp.copy(self.dev_B,self.dev_B_fixed)
        self.norm_max.zero_()
        wp.launch(kernel = Inf_norm,dim=self.N_verts,inputs=[self.dev_B,self.norm_max])
        print('before solve Inf_norm : ',self.norm_max.numpy()[0])
        bsr_mv(self.A,self.dev_delta_x,self.dev_B,alpha=-1.0,beta=1.0)
        # self.squared_sum.zero_()
        # wp.launch(kernel=square_sum,dim=self.dims[layer],inputs=[self.dev_B[layer],self.squared_sum])
        # print('after solve  squared_norm : ',self.squared_sum.numpy()[0])
        self.norm_max.zero_()
        wp.launch(kernel = Inf_norm,dim=self.N_verts,inputs=[self.dev_B,self.norm_max])
        print('after solve Inf_norm : ',self.norm_max.numpy()[0])


    def gradientDescent(self,iterations = 100,lr = 1e-3):
        for step in range(1,iterations+1):
            self.grad_gpu.zero_()
            wp.launch(kernel=compute_partial_elastic_energy_X_noOrder,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.grad_gpu])
            wp.launch(kernel=compute_partial_gravity_energy_X_noOrder,dim=self.N_verts,inputs=[self.m_gpu[0],self.g_gpu,self.grad_gpu])
            wp.launch(kernel=compute_partial_fixed_energy_X_noOrder,dim=self.N_pin,inputs=[self.x_gpu_layer[0],self.pin_list_gpu,self.grad_gpu,self.pin_pos_gpu,self.control_mag])
            if step%(iterations/100) == 0:
                wp.synchronize()
                print('Step : ',step)
                self.energy.zero_()
                wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy])
                wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.m_gpu[0],self.g_gpu,self.pin_gpu,self.all_pin_pos_gpu,self.control_mag,self.energy])
                self.showErrorInfNorm(0,self.grad_gpu)
                print('Energy : ',self.energy.numpy()[0])
                self.plot_x.append(step)
                self.plot_InfNorm.append(self.norm_max.numpy()[0]) 
                self.plot_energy.append(self.energy.numpy()[0])  
            alpha = lr
            for i in range(20):
                wp.launch(kernel=add_grad,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.x_lineSearch,self.grad_gpu,alpha])
                self.energy_lineSearch.zero_()
                wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_lineSearch,self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy_lineSearch])
                wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_lineSearch,self.m_gpu[0],self.g_gpu,self.pin_gpu,self.all_pin_pos_gpu,self.control_mag,self.energy_lineSearch])
                wp.synchronize()
                if self.energy_lineSearch.numpy()[0] < self.energy.numpy()[0]:
                    wp.copy(self.x_gpu_layer[0],self.x_lineSearch)
                    break
                else:
                    alpha = alpha*0.2

    def Newton(self,iterations = 1000):
        for step in range(1,iterations+1):

            self.grad_gpu.zero_()
            wp.copy(self.MF_value_gpu,self.MF_value_fixed_gpu)
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.dev_vertex2index[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.grad_gpu])
            wp.launch(kernel=compute_partial_gravity_energy_X,dim=self.N_verts,inputs=[self.m_gpu[0],self.g_gpu,self.grad_gpu,self.dev_index2vertex[0]])
            wp.launch(kernel=compute_partial_fixed_energy_X,dim=self.N_pin,inputs=[self.x_gpu_layer[0],self.dev_vertex2index[0],self.pin_list_gpu,self.grad_gpu,self.pin_pos_gpu,self.control_mag])
            wp.launch(kernel=compute_elastic_hessian,dim=self.N_hexagons*64,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.MF_value_gpu,self.hex_update_offset_gpu[0]])
            wp.launch(kernel=spd_matrix33f,dim=self.MF_nnz,inputs=[self.MF_value_gpu,self.spd_value])
            if step%(iterations/100) == 0:
                wp.synchronize()
                print('Step : ',step)
                self.energy.zero_()
                wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy])
                wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.m_gpu[0],self.g_gpu,self.pin_gpu,self.all_pin_pos_gpu,self.control_mag,self.energy])
                print('Energy : ',self.energy.numpy()[0])
                self.showErrorInfNorm(0,self.grad_gpu)
                self.plot_x.append(step)
                self.plot_InfNorm.append(self.norm_max.numpy()[0]) 
                self.plot_energy.append(self.energy.numpy()[0])  
            bsr_set_from_triplets(self.L[0],self.MF_L_row_gpu,self.MF_L_col_gpu,self.MF_value_gpu,value_offset=self.off_l)
            bsr_set_from_triplets(self.U[0],self.MF_U_row_gpu,self.MF_U_col_gpu,self.MF_value_gpu,value_offset=self.off_u)
            bsr_set_from_triplets(self.D[0],self.MF_D_row_gpu,self.MF_D_col_gpu,self.MF_value_gpu,value_offset=self.off_d)

            wp.copy(self.dev_B_fixed[0],self.grad_gpu)
            #self.PerformJacobi(layer=0,iterations=3)
            #self.PerformGaussSeidel(layer=0,iterations=1)
            self.PerformConjugateGradient(layer=0,iterations=5)
            

            self.showError(0)
            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.dev_delta_x[0],self.dev_index2vertex[0]])
            #wp.launch(kernel=pin,dim=self.N_pin,inputs=[self.x_gpu_layer[0],self.pin_pos_gpu,self.pin_list_gpu])
    
    def updateNormal(self):
        self.vert_normal_gpu.zero_()
        wp.launch(kernel = updateFaceNorm,dim = self.N_face,inputs = [self.dev_x,self.face_gpu,self.face_normal_gpu])
        wp.launch(kernel= updateVertNorm,dim = self.N_face,inputs = [self.vert_normal_gpu,self.face_gpu,self.face_normal_gpu])


    def VCycle(self,layer):
        self.PerformGaussSeidel(layer,iterations=3)
        self.showError(layer)
        if layer<self.layer-1:
            self.downSample(layer)
            self.VCycle(layer+1)
            self.upSample(layer+1)
        if layer==self.layer-1:
            return 0
        self.PerformGaussSeidel(layer,iterations=3)
        self.showError(layer)



    def NewtonMultigrid(self,iterations = 100):
        for step in range(1,iterations+1):
            self.grad_gpu.zero_()
            wp.copy(self.MF_value_gpu,self.MF_value_fixed_gpu)
            wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.dev_vertex2index[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.grad_gpu])
            wp.launch(kernel=compute_partial_gravity_energy_X,dim=self.N_verts,inputs=[self.m_gpu[0],self.g_gpu,self.grad_gpu,self.dev_index2vertex[0]])
            wp.launch(kernel=compute_partial_fixed_energy_X,dim=self.N_pin,inputs=[self.x_gpu_layer[0],self.dev_vertex2index[0],self.pin_list_gpu,self.grad_gpu,self.pin_pos_gpu,self.control_mag])                    
            #output something for ui
            if step%(iterations/100) == 0:
                wp.synchronize()
                print('Step : ',step)
                self.energy.zero_()
                wp.launch(kernel=compute_elastic_energy,dim=self.N_hexagons*8,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.energy])
                wp.launch(kernel=compute_gravity_energy,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.m_gpu[0],self.g_gpu,self.pin_gpu,self.all_pin_pos_gpu,self.control_mag,self.energy])
                print('Energy : ',self.energy.numpy()[0])
                self.showErrorInfNorm(0,self.grad_gpu)
                self.plot_x.append(step)
                self.plot_InfNorm.append(self.norm_max.numpy()[0]) 
                self.plot_energy.append(self.energy.numpy()[0]) 
            wp.launch(kernel=compute_elastic_hessian,dim=self.N_hexagons*64,inputs=[self.x_gpu_layer[0],self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.MF_value_gpu,self.hex_update_offset_gpu[0]])
            wp.launch(kernel=spd_matrix33f,dim=self.MF_nnz,inputs=[self.MF_value_gpu,self.spd_value])
            #fill matrix
            for i in range(self.layer):
                if i == 0:
                    bsr_set_from_triplets(self.A[i],self.MF_row_gpu,self.MF_col_gpu,self.MF_value_gpu)
                    bsr_set_from_triplets(self.L[i],self.MF_L_row_gpu,self.MF_L_col_gpu,self.MF_value_gpu,value_offset=self.off_l)
                    bsr_set_from_triplets(self.U[i],self.MF_U_row_gpu,self.MF_U_col_gpu,self.MF_value_gpu,value_offset=self.off_u)
                    bsr_set_from_triplets(self.D[i],self.MF_D_row_gpu,self.MF_D_col_gpu,self.MF_value_gpu,value_offset=self.off_d)
                else:
                    self.A[i] = bsr_mm(self.Ut_hat[i-1],bsr_mm(self.A[i-1],self.Us_hat[i-1]))
                    wp.launch(kernel=spd_matrix33f,dim=self.UtAUs_nnz[i-1],inputs=[self.A[i].values,self.spd_value])
                    wp.launch(kernel = block_values_reorder,dim = self.UtAUs_nnz[i-1],inputs=[self.A[i].values,self.UtAUs_values[i-1],self.UtAUs_block_offset_gpu[i-1]])                    
                    bsr_set_from_triplets(self.L[i],self.UtAUs_L_row_gpu[i-1],self.UtAUs_L_col_gpu[i-1],self.UtAUs_values[i-1],value_offset=self.UtAUs_off_l[i-1])
                    bsr_set_from_triplets(self.U[i],self.UtAUs_U_row_gpu[i-1],self.UtAUs_U_col_gpu[i-1],self.UtAUs_values[i-1],value_offset=self.UtAUs_off_u[i-1])
                    bsr_set_from_triplets(self.D[i],self.UtAUs_D_row_gpu[i-1],self.UtAUs_D_col_gpu[i-1],self.UtAUs_values[i-1],value_offset=self.UtAUs_off_d[i-1])

            wp.copy(self.dev_B_fixed[0],self.grad_gpu)
            '''
            #V_cycle  迭代展开3层版本
            self.PerformJacobi(layer=0,iterations=1)
            self.PerformGaussSeidel(layer=0,iterations=1)
            #self.showError(layer=0)

            self.downSample(layer=0)

            self.PerformGaussSeidel(layer=1,iterations=1)
            #self.showError(layer=1)

            self.downSample(layer=1)
            self.PerformGaussSeidel(layer=2,iterations=5)
            #self.showError(layer=2)
            self.upSample(layer=2)
            self.PerformGaussSeidel(layer=1,iterations=1)
            #self.showError(layer=1)

            self.upSample(layer=1)

            self.PerformGaussSeidel(layer=0,iterations=1)
            self.showError(layer=0)
            '''
            self.VCycle(0)
            # print('another')
            # self.VCycle(0)
            self.finish()
           

            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.x_gpu_layer[0],self.dev_x_solved[0],self.dev_index2vertex[0]])
    
    
    # 参考 https://www.math.hkust.edu.hk/~mamu/courses/531/tutorial_with_corrections.pdf   98页的公式
    def FAS(self,iterations = 100):
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
            self.show_layer(1)
            #Compute the coarse-grid approximation to the error
            wp.launch(kernel=z_axpby,dim=self.dims[1],inputs=[self.dev_E[1],self.dev_x_solved[1],self.dev_B[1],-1.0,1.0])
            #Interpolate the error approximation up to the fine grid and correct the current fine-grid approximation
            bsr_mv(self.Us_noOrder_hat[0],self.dev_E[1],self.dev_E[0],alpha=1.0,beta=0.0)
            wp.launch(kernel=axpby,dim=self.dims[0],inputs=[self.dev_E[0],self.dev_x_solved[0],1.0,1.0])
            #self.PerformJacobi(layer=1,iterations=1)
            # self.showError(layer = 1)
            # bsr_mv(self.Us_hat[0],self.dev_delta_x[1],self.dev_delta_x[0],alpha=1.0,beta=0.0)
            # wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.dev_x_solved[0],self.dev_delta_x[0],self.dev_index2vertex[0]])
            # self.x[0] = warp.to_torch(self.dev_x_solved[0]).cpu()
            # self.show_layer(0)

    def select(self,o,d):
        p0 = wp.vec3f(o)
        dir = wp.vec3f(d)
        self.select_num.zero_()
        wp.launch(kernel=selectKernel,dim = self.N_face,inputs=[p0,dir,self.dev_x,self.face_gpu,self.select_num,self.select_faces,self.select_distances])
        wp.synchronize()
        num = self.select_num.numpy()[0]
        if num == 0:
            return False
        else:
            faces = self.select_faces.numpy()
            distances = self.select_distances.numpy()
            self.x_numpy = self.dev_x.numpy()
            idx = -1
            min_dis = 1e10
            for i in range(num):
                if distances[i] < min_dis:
                    min_dis = distances[i]
                    idx = faces[i]
            v0 = self.surface[3*idx+0]
            v1 = self.surface[3*idx+1]
            v2 = self.surface[3*idx+2]
            x0 = self.x_numpy[v0]
            x1 = self.x_numpy[v1]
            x2 = self.x_numpy[v2]
            d0 = Squared_VE_Distance(x0,o,d)
            d1 = Squared_VE_Distance(x1,o,d)
            d2 = Squared_VE_Distance(x2,o,d)
            if d0 < d1 and d0 < d2:
                self.select_vertex = v0
            elif d1 < d2:
                self.select_vertex = v1
            else:
                self.select_vertex = v2
            return True
        
    def moveSelect(self,o,d):
        diff = glm.vec3(self.x_numpy[self.select_vertex])-o
        dist = glm.dot(diff,d)
        self.target = o+dist*d
        wp.launch(kernel= Control_Kernel,dim=self.dims[0],inputs=[self.select_vertex,self.more_fixed_gpu,self.dev_x])

    def clear(self):
        self.select_vertex = -1
        self.more_fixed_gpu.zero_()


    def render(self,pause=False):
        if not pause:
            wp.copy(self.dev_old_x,self.dev_x)
            wp.launch(Basic_Update_Kernel,dim=self.N_verts,inputs=[self.dev_x,self.dev_v,self.damping,self.dt])
            wp.copy(self.dev_inertia_x,self.dev_x)

            self.MF_value_gpu.zero_()
            wp.launch(kernel=Hessian_Mass_Kernel,dim=self.N_verts,inputs=[self.m_gpu,self.MF_value_gpu,self.d_offset_gpu,self.inv_dt])
            wp.launch(kernel=compute_elastic_hessian,dim=self.N_edge,inputs=[self.dev_x,self.edge_gpu,self.rest_length,self.k,self.MF_value_gpu,self.offset_gpu])
            bsr_set_from_triplets(self.A,self.MF_row_gpu,self.MF_col_gpu,self.MF_value_gpu)

            self.grad_gpu.zero_()
            wp.launch(kernel=compute_elastic_gradient,dim=self.N_edge,inputs=[self.dev_x,self.edge_gpu,self.rest_length,self.k,self.grad_gpu])
            wp.launch(kernel=compute_gravity_Gradient,dim=self.N_verts,inputs=[self.m_gpu,self.g,self.grad_gpu])
            wp.launch(kernel=compute_Inertia_Gradient,dim=self.N_verts,inputs=[self.grad_gpu,self.dev_inertia_x,self.dev_x,self.m_gpu,self.inv_dt])
            
            wp.copy(self.dev_B_fixed,self.grad_gpu)
            self.PerformConjugateGradient(iterations=5)
            self.showError()
            wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.dev_x,self.dev_delta_x])
            wp.launch(kernel=updateVelocity,dim=self.N_verts,inputs=[self.dev_x,self.dev_old_x,self.dev_v,self.inv_dt])

            # dir = wp.vec3f()
            # if(self.select_vertex!=-1):
            #     dir = self.target-glm.vec3(self.x_numpy[self.select_vertex])
            #     dir_len = glm.length(dir)
            #     if dir_len > 0.5:
            #         dir_len = 0.5
            #     dir = dir*dir_len
            #     dir = wp.vec3f(dir)
            # wp.launch(kernel= Fixed_Update_Kernel,dim = self.dims[0],inputs = [self.dev_x,self.all_pin_pos_gpu,self.more_fixed_gpu,self.pin_gpu,dir])
            # wp.copy(self.dev_old_x,self.dev_x)
            # wp.launch(Basic_Update_Kernel,dim=self.N_verts,inputs=[self.dev_x,self.dev_v,self.damping,self.dt])
            # wp.copy(self.dev_inertia_x,self.dev_x)


            # self.MF_value_gpu.zero_()
            # wp.launch(kernel=Hessian_Diag_Kernel,dim=self.dims[0],inputs=[self.MF_value_gpu,self.off_d,self.dev_vertex2index[0],self.more_fixed_gpu,self.pin_gpu,self.control_mag])
            # wp.launch(kernel=compute_elastic_hessian,dim=self.N_hexagons*64,inputs=[self.dev_x,self.hexagons_gpu[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.MF_value_gpu,self.hex_update_offset_gpu[0]])
            
            # self.grad_gpu.zero_()
            # wp.launch(kernel=compute_partial_elastic_energy_X,dim=self.N_hexagons*8,inputs=[self.dev_x,self.hexagons_gpu[0],self.dev_vertex2index[0],self.shapeFuncGrad_gpu,self.det_pX_peps_gpu[0],self.inverse_pX_peps_gpu[0],self.IM_gpu,self.LameMu_gpu,self.LameLa_gpu,self.grad_gpu])
            # wp.launch(kernel=compute_partial_gravity_energy_X,dim=self.N_verts,inputs=[self.m_gpu[0],self.g_gpu,self.grad_gpu,self.dev_index2vertex[0]])
            # wp.launch(kernel=compute_partial_fixed_energy_X,dim=self.N_pin,inputs=[self.dev_x,self.dev_vertex2index[0],self.pin_list_gpu,self.grad_gpu,self.pin_pos_gpu,self.control_mag])
            # wp.launch(kernel=compute_partial_more_fixed_energy_X,dim=self.N_verts,inputs=[self.dev_x,self.dev_vertex2index[0],self.more_fixed_gpu,self.grad_gpu,self.all_pin_pos_gpu,self.control_mag])
            # wp.launch(kernel=compute_Inertia_Gradient_Kernel,dim = self.N_verts,inputs = [self.grad_gpu,self.dev_x,self.dev_inertia_x,self.dev_vertex2index[0],self.m_gpu[0],self.inv_dt])

            # bsr_set_from_triplets(self.L[0],self.MF_L_row_gpu,self.MF_L_col_gpu,self.MF_value_gpu,value_offset=self.off_l)
            # bsr_set_from_triplets(self.U[0],self.MF_U_row_gpu,self.MF_U_col_gpu,self.MF_value_gpu,value_offset=self.off_u)
            # bsr_set_from_triplets(self.D[0],self.MF_D_row_gpu,self.MF_D_col_gpu,self.MF_value_gpu,value_offset=self.off_d)

            # wp.copy(self.dev_B_fixed[0],self.grad_gpu)
            # #self.PerformJacobi(layer=0,iterations=3)
            # #self.PerformGaussSeidel(layer=0,iterations=3)
            # self.PerformConjugateGradient(layer=0,iterations=5)
            # self.showError(0)
            
            # wp.launch(kernel=update_deltaX_kernel,dim=self.N_verts,inputs=[self.dev_x,self.dev_delta_x[0],self.dev_index2vertex[0]])  
            # wp.launch(kernel=updateVelocity,dim=self.N_verts,inputs=[self.dev_x,self.dev_old_x,self.dev_v,self.inv_dt])          
            
        self.updateNormal()
        wp.copy(self.render_x,self.dev_x)
        wp.synchronize()
        glDrawElements(GL_TRIANGLES, len(self.surface), GL_UNSIGNED_INT, None)
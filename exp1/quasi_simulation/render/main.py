from OpenGL.GL import *
from OpenGL.GL import shaders
from window import Window	# 第一章中封装的Window类
import numpy as np
import warp as wp
from cuda import cudart

def check_cudart_err(args):
    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(format_cudart_err(err))

    return ret


wp.init()
# 创建窗口
w = Window(1920, 1080, "Test")
# 定义数据
triangle = np.array([
    -0.5, -0.5, 0, 1, 0, 0,
    0.5, -0.5, 0, 0, 1, 0,
    0, 0.5, 0, 0, 0, 1
], dtype=np.float32)
tri_gpu= wp.from_numpy(triangle)
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, tri_gpu.size*4, None, GL_DYNAMIC_DRAW)
graphics_ressource = check_cudart_err(cudart.cudaGraphicsGLRegisterBuffer( VBO, cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))
check_cudart_err(cudart.cudaGraphicsMapResources(1, graphics_ressource,None))
ptr, size = check_cudart_err(cudart.cudaGraphicsResourceGetMappedPointer(graphics_ressource))
vert_array =wp.array(ptr=ptr,length=tri_gpu.size/3,shape =None,dtype=wp.vec3f,device='cuda:0')

#wp.copy(vert_array,tri_gpu)

triangle = np.array([
    -0.5, -0.5, -1, 1, 0, 0,
    0.5, -0.5, -1, 0, 0, 1,
    0, 0.5, -1, 0, 0, 1
], dtype=np.float32)
tri_gpu= wp.from_numpy(triangle,dtype=wp.vec3f)
wp.copy(vert_array,tri_gpu)
wp.synchronize()
#print(vert_array)

# 创建、绑定VBO

# 着色器（具体内容在前面）
# 顶点着色器
vs = """
#version 330 core
in vec3 aPos;
in vec3 aColor;

out vec3 VertexColor;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
    VertexColor = aColor;
}
"""

# 片元着色器
fs = """
#version 330 core
in vec3 VertexColor;

out vec4 FragColor;

void main()
{
    FragColor = vec4(VertexColor.rgb, 1.0f);
} 
"""

# 编译着色器
vsProgram = shaders.compileShader(vs, GL_VERTEX_SHADER)
fsProgram = shaders.compileShader(fs, GL_FRAGMENT_SHADER)
program = shaders.compileProgram(vsProgram, fsProgram)
# 解释数据含义
aPosLoc = glGetAttribLocation(program, 'aPos')
glVertexAttribPointer(aPosLoc, 3, GL_FLOAT, False, 24, ctypes.c_void_p(0))
glEnableVertexAttribArray(aPosLoc)
aColorLoc = glGetAttribLocation(program, 'aColor')
glVertexAttribPointer(aColorLoc, 3, GL_FLOAT, False, 24, ctypes.c_void_p(12))
glEnableVertexAttribArray(aColorLoc)
# 渲染循环
def render():
    glDrawArrays(GL_TRIANGLES, 0, 3)

w.loop(render)


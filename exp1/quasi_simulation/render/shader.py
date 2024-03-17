from OpenGL.GL import shaders
from OpenGL.GL import *
import numpy as np

class Shader:
    def __init__(self, vsPath, fsPath) -> None:
        """ 读取 GLSL 文件并编译 """
        with open(vsPath, 'r',encoding='utf-8') as f:
            text = f.read()
            vs = shaders.compileShader(text, GL_VERTEX_SHADER)

        with open(fsPath, 'r',encoding='utf-8') as f:
            text = f.read()
            fs = shaders.compileShader(text, GL_FRAGMENT_SHADER)

        self.shader = shaders.compileProgram(vs, fs)

    def use(self):
        glUseProgram(self.shader)

    def setUniform(self, name, value):
        """ 根据传入的数据类型，自动选择 glUniformX 函数 """
        self.use()
        loc = glGetUniformLocation(self.shader, name)
        dtype = type(value)
        if dtype == np.ndarray:
            size = value.size
            dtype = value.dtype
            funcs = {
                np.int32: [glUniform1i, glUniform2i, glUniform3i, glUniform4i],
                np.uint: [glUniform1ui, glUniform2ui, glUniform3ui, glUniform4ui],
                np.float32: [glUniform1f, glUniform2f, glUniform3f, glUniform4f],
                np.double: [glUniform1d, glUniform2d, glUniform3d, glUniform4d],
            }
            func = funcs[dtype][size - 1]
            func(loc, *value)
            return
        elif dtype == int or dtype == np.int32 or dtype == np.int64:
            glUniform1i(loc, value)
        elif dtype == float or dtype == np.float64 or dtype == np.float32:
            glUniform1f(loc, value)
        else:
            raise RuntimeError("未知的参数类型！")

    def setAttrib(self, name, size, dtype, stride, offset):
        """ 设置顶点属性链接 """
        loc = glGetAttribLocation(self.shader, name)
        glVertexAttribPointer(loc, size, dtype, False, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(loc)

    def setMatrix(self,name,value):
        loc = glGetUniformLocation(self.shader, name)
        glUniformMatrix4fv(loc,1,GL_FALSE,value)

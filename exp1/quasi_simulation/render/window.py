from OpenGL.GL import *	# 导入PyOpenGL
import glfw				# 导入GLFW
from render.shader import Shader
from render.camera import Camera
import glm
import numpy as np

class MOTION():
    NO_MOTION=0
    ZOOM_MOTION = 1
    ROTATE_MOTION=2
    TRANSLATE_MOTION=3

class Window:
    def __init__(self, width, height, title, bgColor=(1.0, 1.0, 1.0, 0.0)):
        # 初始化GLFW
        if not glfw.init():
            raise RuntimeError("GLFW初始化失败！")
        # 创建窗口
        self.width, self.height, self.title, self.bgColor = width, height, title, bgColor
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self.window = glfw.create_window(width, height, title, None, None)
        # 显示窗口
        self.show()
        #self.shader = Shader("assets/shaders/base.vert", "assets/shaders/base.frag")
        self.phongShader = Shader("assets/shaders/phong.vert", "assets/shaders/phong.frag")
        self.shadowShader = Shader("assets/shaders/shadow.vert", "assets/shaders/shadow.frag")

        self.motion_mode = MOTION.NO_MOTION
        self.pause = True
        self.camera = Camera(glm.vec3(0.0, 0.0, 5.0), glm.vec3(0.0, 0.0, -1.0), glm.vec3(0.0, 1.0, 0.0))
        glfw.set_mouse_button_callback(self.window, self.mouse_click_callback)
        glfw.set_cursor_pos_callback(self.window,self.mouse_move_callback)

        self.modelMatrix = glm.mat4(1.0)
        self.projMatrix = glm.perspective(glm.radians(45.0),float(self.width)/float(self.height),0.1,100.0)
        self.lightPos = glm.vec3(-2.0, 2.0, 0.0)
        self.phongShader.setVec3("light_position", self.lightPos.to_tuple()) 
        
    def show(self):
        glfw.make_context_current(self.window)
        glfw.set_window_size_limits(self.window, self.width, self.height, self.width, self.height)
        glViewport(0, 0, self.width, self.height)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)

    def loop(self, render):	
        while not glfw.window_should_close(self.window):
            glClearColor(*self.bgColor)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


            self.phongShader.use()
            self.camera.update()
            self.phongShader.setMatrix('_modelMatrix',self.modelMatrix.to_tuple())
            self.phongShader.setMatrix('_viewMatrix',self.camera.getMatrix().to_tuple())
            self.phongShader.setMatrix('_projMatrix',self.projMatrix.to_tuple())
            
            render(self.pause)	

            glfw.swap_buffers(self.window)
            glfw.poll_events()
            self.ProcessKeyInput()

        glfw.destroy_window(self.window)
        glfw.terminate()
    
    def ProcessKeyInput(self):
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            self.pause = not self.pause
    
    def mouse_move_callback(self, window, xpos, ypos):
        if self.motion_mode == MOTION.TRANSLATE_MOTION:
            self.camera.move(self.mouse_x-xpos, self.mouse_y-ypos)
        elif self.motion_mode == MOTION.ZOOM_MOTION:
            self.camera.zoom(ypos-self.mouse_y)
        self.mouse_x = xpos
        self.mouse_y = ypos

    def mouse_click_callback(self, window, button:int, action:int, mods:int):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            self.motion_mode = MOTION.NO_MOTION
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:                
                self.motion_mode = MOTION.TRANSLATE_MOTION
            elif glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
                self.motion_mode = MOTION.ZOOM_MOTION
            else :
                self.motion_mode = MOTION.ROTATE_MOTION
            xpos, ypos = glfw.get_cursor_pos(window)
            self.mouse_x = xpos
            self.mouse_y = ypos
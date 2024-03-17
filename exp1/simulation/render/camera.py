import glm

class Camera:
    def __init__(self, position, front, up):
        self.position = position
        self.initPosition = position
        self.front = front
        self.up = up
        self.view = glm.lookAt(position, front, up)
        self.speed = 0.01
        self.pitch = 0.0
        self.yaw = -90.0
        self.sensitivity = 0.05
        self.xpos = 0.0
        self.ypos = 0.0

        self.firstMouse = True
        self.vMatrix = glm.mat4(1.0)

    def setSpeed(self, speed):
        self.speed = speed

    def lookAt(self, pos,front,up):
        self.initPosition = pos
        self.position = pos
        self.front = glm.normalize(front)
        self.up = up
        self.vMatrix = glm.lookAt(self.position, self.position + self.front, self.up)

    def update(self):
        self.vMatrix = glm.lookAt(self.position, self.position + self.front, self.up)

    def getMatrix(self):
        return self.vMatrix
    
    def zoom(self,yOffset):
        self.position +=0.05*yOffset

    def move(self, xOffset,yOffset):
        self.position += 0.005*xOffset* glm.normalize(glm.cross(self.front, self.up))
        self.position -= 0.005*yOffset* self.up
        # if mode == CAMERA_MOVE.MOVE_LEFT:
        #     self.position -= self.speed * glm.normalize(glm.cross(self.front, self.up))
        #     return
        # elif mode == CAMERA_MOVE.MOVE_RIGHT:
        #     self.position += self.speed * glm.normalize(glm.cross(self.front, self.up))
        #     return
        # elif mode == CAMERA_MOVE.MOVE_FRONT:
        #     self.position += self.speed * self.front
        #     return
        # elif mode == CAMERA_MOVE.MOVE_BACK:
        #     self.position -= self.speed * self.front
        #     return
        # elif mode == CAMERA_MOVE.MOVE_UP:
        #     self.position += self.speed * self.up
        #     return
        # elif mode == CAMERA_MOVE.MOVE_DOWN:
        #     self.position -= self.speed * self.up
        #     return
        # elif mode == CAMERA_MOVE.MOVE_InitialPosition:
        #     self.position = self.initPosition
        #     return
        
    def yaw(self,xOffset):
        self.yaw += xOffset * self.sensitivity
        self.front.y = glm.sin(glm.radians(self.pitch))
        self.front.x = glm.cos(glm.radians(self.pitch)) * glm.cos(glm.radians(self.yaw))
        self.front.z = glm.cos(glm.radians(self.pitch)) * glm.sin(glm.radians(self.yaw))
        self.front = glm.normalize(self.front)
        self.update()

    def pitch(self,yOffset):
        self.pitch += yOffset * self.sensitivity
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0
        self.front.y = glm.sin(glm.radians(self.pitch))
        self.front.x = glm.cos(glm.radians(self.pitch)) * glm.cos(glm.radians(self.yaw))
        self.front.z = glm.cos(glm.radians(self.pitch)) * glm.sin(glm.radians(self.yaw))
        self.front = glm.normalize(self.front)
        self.update()

    def setSensitive(self, sensitivity):
        self.sensitivity = sensitivity

    def onMouseMove(self, xpos, ypos):
        if self.firstMouse:
            self.xpos = xpos
            self.ypos = ypos
            self.firstMouse = False
            return
        xoffset = xpos - self.xpos
        yoffset = self.ypos - ypos
        self.pitch(yoffset)
        self.yaw(xoffset)        
        self.xpos = xpos
        self.ypos = ypos
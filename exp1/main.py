import taichi as ti
import math
import pyvista as pv
import numpy as np
ti.init(arch=ti.cpu)

# global control
paused = True

N = 0
N_edges = 0
N_hexagons = 0

mesh_path = 'assets/objs/dragon.obj'
mesh = pv.read(mesh_path)
voxels = pv.voxelize(mesh, density=0.3, check_surface=False)

N = voxels.points.shape[0]

hex = []
edge = []
for cell in voxels.cell:
    N_hexagons+=1
    N_edges+=12
    hex.append([cell.point_ids[0],cell.point_ids[4],cell.point_ids[3],cell.point_ids[7],cell.point_ids[1],cell.point_ids[5],cell.point_ids[2],cell.point_ids[6]])
    edge.append([cell.point_ids[0],cell.point_ids[1]])
    edge.append([cell.point_ids[1],cell.point_ids[2]])
    edge.append([cell.point_ids[2],cell.point_ids[3]])
    edge.append([cell.point_ids[3],cell.point_ids[0]])
    edge.append([cell.point_ids[4],cell.point_ids[5]])
    edge.append([cell.point_ids[5],cell.point_ids[6]])
    edge.append([cell.point_ids[6],cell.point_ids[7]])
    edge.append([cell.point_ids[7],cell.point_ids[4]])
    edge.append([cell.point_ids[0],cell.point_ids[4]])
    edge.append([cell.point_ids[1],cell.point_ids[5]])
    edge.append([cell.point_ids[2],cell.point_ids[6]])
    edge.append([cell.point_ids[3],cell.point_ids[7]])


dx = 1/32
curser_radius = dx/2

# physical quantities
m = 1.0
g = 9.8
YoungsModulus = ti.field(ti.f32, ())
YoungsModulus[None] = 1e6
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# substepping
substepping = 100
# time-step size (for time integration)
dh = h/substepping


# simulation components
x = ti.Vector.field(3, ti.f32, N)


v = ti.Vector.field(3, ti.f32, N)
total_energy = ti.field(ti.f32, (), needs_grad=False)
grad = ti.Vector.field(3, ti.f32, N)

# geometric components
hexagons = ti.Vector.field(8, ti.i32, N_hexagons)



# -----------------------meshing and init----------------------------

def meshing():
    for i in range(N):
        x[i][0] = voxels.points[i][0]
        x[i][1] = voxels.points[i][1]
        x[i][2] = voxels.points[i][2]
        v[i] = ti.Vector([0.0, 0.0,0.0])
    
    for i in range(N_hexagons):
        for j in range(8):
            hexagons[i][j] = hex[i][j]



shapeFuncGrad = ti.Vector.field(3,ti.f32,(2,2,2,2,2,2))

help = ti.Vector.field(3,ti.f32,8)
help[0] = ti.Vector([-1,-1,-1])
help[1] = ti.Vector([-1,-1,1])
help[2] = ti.Vector([-1,1,-1])
help[3] = ti.Vector([-1,1,1])
help[4] = ti.Vector([1,-1,-1])
help[5] = ti.Vector([1,-1,1])
help[6] = ti.Vector([1,1,-1])
help[7] = ti.Vector([1,1,1])

quadrature = ti.Vector.field(3,ti.f32,8)
quadrature[0] = ti.Vector([-0.57735,-0.57735,-0.57735])
quadrature[1] = ti.Vector([-0.57735,-0.57735,0.57735])
quadrature[2] = ti.Vector([-0.57735,0.57735,-0.57735])
quadrature[3] = ti.Vector([-0.57735,0.57735,0.57735])
quadrature[4] = ti.Vector([0.57735,-0.57735,-0.57735])
quadrature[5] = ti.Vector([0.57735,-0.57735,0.57735])
quadrature[6] = ti.Vector([0.57735,0.57735,-0.57735])
quadrature[7] = ti.Vector([0.57735,0.57735,0.57735])

@ti.kernel
def calShapeFuncGrad():
    for i in range(2):
        for j in range(2):
            for k in range(2):
                whichShapeFunc = i*4+j*2+k
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            whichQuadrature = 4*a+2*b+c
                            shapeFuncGrad[i,j,k,a,b,c][0] = help[whichShapeFunc][0]*(1+help[whichShapeFunc][1]*quadrature[whichQuadrature][1])*(1+help[whichShapeFunc][2]*quadrature[whichQuadrature][2])*0.125
                            shapeFuncGrad[i,j,k,a,b,c][1] = help[whichShapeFunc][1]*(1+help[whichShapeFunc][0]*quadrature[whichQuadrature][0])*(1+help[whichShapeFunc][2]*quadrature[whichQuadrature][2])*0.125
                            shapeFuncGrad[i,j,k,a,b,c][2] = help[whichShapeFunc][2]*(1+help[whichShapeFunc][1]*quadrature[whichQuadrature][1])*(1+help[whichShapeFunc][0]*quadrature[whichQuadrature][0])*0.125

#for general hex,we need patial X/patial epsilon on the 8 quadrature points . Often we use the inverse of the matrix,so we save the inverse.
inverse_pX_peps = ti.Matrix.field(3,3,ti.f32,(N_hexagons,8))
det_pX_peps = ti.field(ti.f32,(N_hexagons,8))
avg_det_pX_peps = ti.field(ti.f32,N_hexagons)
F = ti.Matrix.field(3, 3, ti.f32,N_hexagons)
IM = ti.Matrix([[1,0,0],[0,1,0],[0,0,1]],dt=ti.f32)
@ti.kernel
def prepare():
    for hex in range(N_hexagons):
        count_quadrature = 0
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for row in ti.static(range(3)):
                        for col in ti.static(range(3)):
                            value_now = 0.0
                            for i in ti.static(range(2)):
                                for j in ti.static(range(2)):
                                    for k in ti.static(range(2)):
                                        value_now += x[hexagons[hex][i*4+j*2+k]][row]*shapeFuncGrad[i,j,k,a,b,c][col]
                            F[hex][row,col] = value_now
                    det_pX_peps[hex,count_quadrature] = F[hex].determinant()
                    inverse_pX_peps[hex,count_quadrature] = F[hex].inverse()
                    count_quadrature += 1
        avg_det_pX_peps[hex] = 0.0
        for i in range(8):
            avg_det_pX_peps[hex] += det_pX_peps[hex,i]
        avg_det_pX_peps[hex] *= 0.125


@ti.kernel
def compute_F():
    for i in range(N):
        grad[i].fill(0.0)

    for hex in range(N_hexagons):
        count_quadrature = 0

        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for row in ti.static(range(3)):
                        for col in ti.static(range(3)):
                            value_now = 0.0
                            for i in ti.static(range(2)):
                                for j in ti.static(range(2)):
                                    for k in ti.static(range(2)):
                                        value_now += x[hexagons[hex][i*4+j*2+k]][row]*shapeFuncGrad[i,j,k,a,b,c][col]
                            F[hex][row,col] = value_now
                    F[hex] = F[hex] @ inverse_pX_peps[hex,count_quadrature]
                    E = 0.5*(F[hex].transpose()@F[hex]-IM)
                    P = F[hex] @ (2*LameMu[None]*E + LameLa[None]*E.trace()*IM)
                    for i in ti.static(range(2)):
                        for j in ti.static(range(2)):
                            for k in ti.static(range(2)):
                                shapeFuncGradNow = ti.Matrix([[shapeFuncGrad[i,j,k,a,b,c][0]],[shapeFuncGrad[i,j,k,a,b,c][1]],[shapeFuncGrad[i,j,k,a,b,c][2]]],dt=ti.f32)
                                temAns = P@(inverse_pX_peps[hex,count_quadrature].transpose())@shapeFuncGradNow*det_pX_peps[hex,count_quadrature]
                                grad[hexagons[hex][i*4+j*2+k]] += temAns
                    count_quadrature+=1
    #for i in range(N):
    #    print(grad[i])    

@ti.kernel
def update():
    # perform time integration
    for i in range(N):
        #fixed point
        if i == 0:
            continue
        # symplectic integration 
        acc = -grad[i]/m - ti.Vector([0.0, g,0.0])
        v[i] += dh*acc
        x[i] += dh*v[i]







@ti.kernel
def updateLameCoeff():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))

def sub_stVK():
    for i in range(substepping):
        compute_F()
        update()

# init once and for all
meshing()
calShapeFuncGrad()
prepare()
updateLameCoeff()




####   render components

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *



x_old = 0
y_old = 0


pos_x:float = -0.5
pos_y:float = -0.5
pos_z:float = -10.0
angle_x:float = 0.0
angle_y:float = 0.0

zoom_per_scroll:float = -pos_z/10.0

current_scroll:int = 5

is_holding_mouse = False
is_update = False

def idle():
    if True:
        sub_stVK()
        glutPostRedisplay()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(pos_x, pos_y, pos_z)
    glRotatef(angle_x, 1.0, 0.0, 0.0)
    glRotatef(angle_y, 0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    for i in range(len(edge)):
        face = edge[i]
        glVertex3f(x[face[0]][0],x[face[0]][1],x[face[0]][2])
        glVertex3f(x[face[1]][0],x[face[1]][1],x[face[1]][2])
    glEnd()
    glutSwapBuffers()

def timer(value):
    if is_update:
        is_update = False
        glutPostRedisplay()
    glutTimerFunc(15, timer, 0)

def mouse(button, state, x, y):
    global x_old, y_old, is_holding_mouse, pos_x, pos_y, pos_z, current_scroll,is_update
    is_update = True
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            x_old = x
            y_old = y
            is_holding_mouse = True
        else:
            is_holding_mouse = False
    if state == GLUT_UP:
        if button == 3:
            if current_scroll>0:
                current_scroll -= 1
                pos_z += zoom_per_scroll
                glutPostRedisplay()
        elif button == 4:
            if current_scroll<15:
                current_scroll += 1
                pos_z -= zoom_per_scroll
                glutPostRedisplay()

def motion(x, y):
    global x_old, y_old, is_holding_mouse, pos_x, pos_y, pos_z, angle_x, angle_y,is_update
    if is_holding_mouse:
        is_update = True
        angle_y+= (x - x_old)
        x_old = x
        if(angle_y>360.0):
            angle_y-=360.0
        elif(angle_y<-360.0):
            angle_y+=360.0

        angle_x+= (y - y_old)
        y_old = y
        if(angle_x>360.0):
            angle_x-=360.0
        elif(angle_x<-360.0):
            angle_x+=360.0
        glutPostRedisplay()

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 800)
    glutCreateWindow(b"Triangle")

    glClearColor(0.4, 0.4, 0.4, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(20.0, 1.0, 1.0, 2000.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glLineWidth(1.0)
    glutDisplayFunc(display)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutIdleFunc(idle)
    glutMainLoop()

if __name__ == "__main__":
    main()

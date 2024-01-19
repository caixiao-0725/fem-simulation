from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import trimesh

mesh_path = '../assets/objs/dragon.obj'
mesh = trimesh.load_mesh(mesh_path)
vertices = mesh.vertices
edge = mesh.edges


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(-0.5, -0.5, -10.0)
    glBegin(GL_LINES)
    for i in range(len(edge)):
        face = edge[i]
        glVertex3fv(vertices[face[0]])
        glVertex3fv(vertices[face[1]])
    glEnd()

    glutSwapBuffers()

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)
    glutInitWindowSize(800, 800)
    glutCreateWindow(b"Triangle")
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(20.0, 1.0, 1.0, 2000.0)
    glMatrixMode(GL_MODELVIEW)
    glLineWidth(1.0)
    glutDisplayFunc(display)
    glutMainLoop()

if __name__ == "__main__":
    main()
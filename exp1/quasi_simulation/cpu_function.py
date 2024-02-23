import numpy as np
import torch
def calShapeFuncGrad(shapeFuncGrad,help,quadrature):
    for whichShapeFunc in range(8):
        for whichQuadrature in range(8):
            shapeFuncGrad[whichShapeFunc][whichQuadrature][0]=help[whichShapeFunc][0]*(1+help[whichShapeFunc][1]*quadrature[whichQuadrature][1])*(1+help[whichShapeFunc][2]*quadrature[whichQuadrature][2])*0.125
            shapeFuncGrad[whichShapeFunc][whichQuadrature][1]=help[whichShapeFunc][1]*(1+help[whichShapeFunc][0]*quadrature[whichQuadrature][0])*(1+help[whichShapeFunc][2]*quadrature[whichQuadrature][2])*0.125
            shapeFuncGrad[whichShapeFunc][whichQuadrature][2]=help[whichShapeFunc][2]*(1+help[whichShapeFunc][1]*quadrature[whichQuadrature][1])*(1+help[whichShapeFunc][0]*quadrature[whichQuadrature][0])*0.125


def ijk_index(point, origin, spacing):
    return ((point - origin) / spacing).int().tolist()

def color_ind(point, origin, spacing):
    temp = ((point - origin) / spacing + +torch.tensor([0.5,0.5,0.5])).int()
    temp[0] = temp[0] % 2
    temp[1] = temp[1] % 2
    temp[2] = temp[2] % 2
    return temp[0] + 2 * temp[1] + 4 * temp[2]
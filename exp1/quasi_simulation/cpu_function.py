import numpy as np

def ijk_index(point, origin, spacing):
    return ((point - origin) / spacing).int().tolist()
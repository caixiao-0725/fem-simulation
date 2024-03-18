from object import Object
import warp as wp
import glm

from render.window import Window
if __name__ == '__main__':
    wp.init()
    wp.set_device('cuda:0')
    
    win = Window(960, 540, "Dynamic Simulation")
    
    pinned = [] #184  732
    obj = Object('assets/objs/dragon.obj',0.05,pinned)

    win.setSelect(obj.select,obj.moveSelect,obj.clear)
    win.loop(obj.render)


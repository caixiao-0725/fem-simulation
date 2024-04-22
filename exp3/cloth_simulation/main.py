from object import Cloth
import warp as wp
import glm

from render.window import Window
if __name__ == '__main__':
    wp.init()
    wp.set_device('cuda:0')
    
    win = Window(960, 540, "Dynamic Simulation")
    
    obj = Cloth()

    win.setSelect(obj.select,obj.moveSelect,obj.clear)
    #win.loop(obj.render)
    obj.train()

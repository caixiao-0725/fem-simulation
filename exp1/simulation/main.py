from object import Object
import warp as wp

from render.window import Window
if __name__ == '__main__':
    wp.init()
    wp.set_device('cuda:0')
    
    win = Window(1920, 1080, "Test")
    
    pinned = [] #184  732
    obj = Object('assets/objs/dragon.obj',0.05,pinned)

    win.loop(obj.render)


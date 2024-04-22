from object import Object
import warp as wp

from render.window import Window
if __name__ == '__main__':
    wp.init()
    wp.set_device('cuda:0')
    
    win = Window(20, 80, "Test")
    
    pinned = [1] #184  732
    obj = Object('assets/objs/dragon.obj',0.5,pinned)

    
    obj.Adam(10000)
    #obj.train(1000)
    obj.show_layer(0)
    #win.loop(obj.render)


from object import Object
import warp as wp

from render.window import Window
if __name__ == '__main__':
    wp.init()
    wp.set_device('cuda:0')
    
    win = Window(20, 10, "Test")
    
    pinned = [732] #184  732
    obj = Object('assets/objs/dragon.obj',0.05,pinned)
    #obj.train(1000)
    obj.compare(5)



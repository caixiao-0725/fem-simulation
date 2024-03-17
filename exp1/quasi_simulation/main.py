from object import Object
import warp as wp

from render.window import Window
if __name__ == '__main__':
    wp.init()
    wp.set_device('cuda:0')
    
    win = Window(1920, 1080, "Test")
    
    pinned = [732] #184  732
    obj = Object('assets/objs/dragon.obj',0.05,pinned)

    #win.loop(obj.render)
    #obj.show_layer(0)
    #obj.show_layer(1)
    #obj.show_layer(2)
    #obj.NewtonMultigrid(1000)
    #obj.Newton(10000)
    obj.FAS(10)
    #obj.MG_compare(10)
    obj.show_layer(0)
    #obj.Adam(iterations=5000)
    #obj.gradientDescent(10000,1.0)
    #obj.show()

    #obj.compare(1)

    #obj.drag()
    #obj.show()
    #obj.show_layer(0)


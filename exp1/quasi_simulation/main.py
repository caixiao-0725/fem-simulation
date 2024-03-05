from object import Object
import warp as wp
if __name__ == '__main__':
    wp.init()
    wp.set_device('cuda:0')
    pinned = [] #184  732
    obj = Object('assets/objs/dragon.obj',0.05,pinned)
    #obj.show_layer(0)
    #obj.show_layer(1)
    #obj.show_layer(2)
    obj.NewtonMultigrid(100)
    #obj.Newton(100)
    #obj.Adam(iterations=5000)
    #obj.gradientDescent(10000,1.0)
    
    obj.show()

    #obj.compare(15000)
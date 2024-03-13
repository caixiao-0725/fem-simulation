from object import Object
import warp as wp
if __name__ == '__main__':
    wp.init()
    wp.set_device('cuda:0')
    pinned = [732] #184  732
    obj = Object('assets/objs/dragon.obj',0.05,pinned)
    #obj.show_layer(0)
    #obj.show_layer(1)
    #obj.show_layer(2)
    #obj.NewtonMultigrid(10)
    #obj.Newton(10)
    #obj.FAS(1)
    obj.MG_compare(1)
    #obj.show_layer(1)
    #obj.Adam(iterations=5000)
    #obj.gradientDescent(10000,1.0)
    #obj.show()

    #obj.compare(1)

    #obj.drag()
    #obj.show()
    #obj.show_layer(0)
from object import Object
import warp as wp
if __name__ == '__main__':
    wp.init()
    wp.set_device('cuda:0')
    pinned = [732] #184
    obj = Object('assets/objs/dragon.obj',0.05,pinned)
    #obj.show_layer(0)
    #obj.show_layer(1)
    #obj.show_layer(2)
    #obj.NewtonMultigrid(1000)
    obj.Newton(10000)
    #obj.gradientDescent(iterations=10000,lr=10.0)
    
    obj.show()
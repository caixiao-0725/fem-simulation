from object import Object
import warp as wp
if __name__ == '__main__':
    wp.init()
    wp.set_device('cuda:0')
    pinned = [732]
    obj = Object('assets/objs/dragon.obj',0.05,pinned)
    obj.Newton()
    #obj.gradientDescent(iterations=10000,lr=10.0)
    obj.show()
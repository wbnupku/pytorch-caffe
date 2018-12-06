from caffenet import *



class args(object):
    pass


args.height = 224
args.width = 224
args.meanB = 0
args.meanG = 0
args.meanR = 0
args.scale = 1

def load_image(imgfile):
    import caffe
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, args.height, args.width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
    transformer.set_raw_scale('data', args.scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, args.height, args.width)
    return image

def forward_pytorch(protofile, weightfile, image):
    net = CaffeNet(protofile)
    print(net)
    net.load_weights(weightfile)
    net.eval()
    image = torch.from_numpy(image)
    image = Variable(image)
    blobs = net(image)
    return blobs, net.models

imgfile = 'data/cat.jpg'
protofile = 'models/resnet50-20sku-20181122/ResNet-50-deploy.prototxt'
weightfile = 'models/resnet50-20sku-20181122/resnet50_20sku-feature.caffemodel'
image = load_image(imgfile)
pytorch_blobs, pytorch_models = forward_pytorch(protofile, weightfile, image)

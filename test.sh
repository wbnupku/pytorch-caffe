


# python verify_deploy.py --protofile models/resnet50-20sku-20181122/ResNet-50-deploy.prototxt  --weightfile models/resnet50-20sku-20181122/resnet50_20sku-feature.caffemodel   --imgfile data/cat.jpg --meanB 104.01 --meanG 116.67 --meanR 122.68 --scale 255 --height 224 --width 224 --synset_words data/synset_words.txt --cuda


# python verify_time.py --protofile models/resnet50-20sku-20181122/ResNet-50-deploy.prototxt --weightfile models/resnet50-20sku-20181122/resnet50_20sku-feature.caffemodel --imgfile data/cat.jpg --meanB 104.01 --meanG 116.67 --meanR 122.68 --scale 255 --height 224 --width 224 --synset_words data/synset_words.txt # --cuda



python verify_deploy.py --protofile models/cls8w/deploy.prototxt --weightfile models/cls8w/deploy.caffemodel  --imgfile data/Koala.jpg --meanB 117 --meanG 117 --meanR 117 --scale 255 --height 224 --width 224 --synset_words models/cls8w/label.txt --cuda
# python verify_time.py --protofile models/cls8w/deploy.prototxt --weightfile models/cls8w/deploy.caffemodel  --imgfile data/Koala.jpg --meanB 117 --meanG 117 --meanR 117 --scale 255 --height 224 --width 224 --synset_words models/cls8w/label.txt --cuda

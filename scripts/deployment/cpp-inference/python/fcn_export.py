#!/usr/bin/env python

import os
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import gluoncv
from gluoncv.model_zoo import FCN
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from PIL import Image

ctx = mx.cpu(0)

# Recreate the model
num_classes = 21
net = FCN(nclass=num_classes, backbone='resnet50', crop_size=240)

# Load the model parameters
model_file='model/model_best.params'
net.load_parameters(model_file, ctx)

# Hybridize the model
net.hybridize()

# Load test image as in https://gluon-cv.mxnet.io/build/examples_segmentation/demo_fcn.html
os.system('wget -O test.jpg https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/1.jpg')
image_file='test.jpg'

with open(image_file, "rb") as f:
    img = f.read()
    b = bytearray(img)

im = Image.open(image_file)
plt.imshow(im)
plt.axis('off')

img = mx.image.imdecode(b)

transform_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
img = transform_fn(img)
img = img.expand_dims(0).as_in_context(ctx)

# Inference with recreated model
# Forward pass required for exporting
# See: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/save_load_params.html
pred = net.demo(img)
data = mx.nd.softmax(pred, 1)
pred = np.argmax(np.squeeze(data), axis=0).asnumpy().astype('uint8')
plt.imshow(pred, vmin=0, vmax=num_classes-1, cmap='jet')
plt.show()

# Export the model (parameters and network)
net.export('fcn_exported_model')

# Import the exported model
deserialized_net = gluon.nn.SymbolBlock.imports(
    "fcn_exported_model-symbol.json",
    ['data'],
    "fcn_exported_model-0000.params", ctx=ctx)

# Inference with exported model
pred = deserialized_net(img)
data = mx.nd.softmax(pred[0], 1)
pred = np.argmax(np.squeeze(data), axis=0).asnumpy().astype('uint8')
plt.imshow(pred, vmin=0, vmax=num_classes-1, cmap='jet')
plt.show()


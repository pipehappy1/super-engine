import yaml
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from theano.tensor.signal.pool import pool_2d
from mlbase.util import floatX
import mlbase.init as winit


class Layer(yaml.YAMLObject):
    
    debugname = 'update layer name'
    LayerTypeName = 'Layer'
    yaml_tag = u'!Layer'
    
    def __init__(self):
        # Layer name may used to print/debug
        # per instance
        self.name = 'Layer'
        # Layer name may used for saving
        # per instance
        self.saveName = 'saveName'
        
        # layer may have multiple input/output
        # only used for network
        # should not access directly
        self.inputLayer = []
        self.outputLayer = []
        self.inputLayerName = []
        self.outputLayerName = []
    
    def getpara(self):
        """
        Parameter collected from here will all updated by gradient.
        """
        return []

    def getExtraPara(self, inputtensor):
        """
        Parameters that are not in the collection for updating by backpropagation.
        """
        return []
    
    def forward(self, inputtensor):
        """
        forward link used in training

        inputtensor: a tuple of theano tensor

        return: a tuple of theano tensor
        """
        return inputtensor

    """
    Use the following code to define
    the layer which may be different
    from the one used in training.

    def predictForward(self, inputtensor):
        return inputtensor

    One example would be batch normalization
    to implement this interface.
    """
    predictForward = forward

    
    def forwardSize(self, inputsize):
        """
        Get output size based on input size.
        For one layer, the input and output size may
        have more than one connection.

        inputsize: A list of tuple of int
        
        return: A list of tuple of int
        """
        return inputsize

    def fillToObjMap(self):
        """
        Return a mapping representing the object
        and the mapping is for YAML dumping.
        """
        objDict = {
            'name': self.name,
            'saveName': self.saveName,
            'inputLayerName': [layer.saveName for layer in self.inputLayer],
            'outputLayerName': [layer.saveName for layer in self.outputLayer]
        }
        return objDict

    def loadFromObjMap(self, tmap):
        """
        Fill the object from mapping tmap
        and used to load the object from YAML dumping.
        """
        self.name = tmap['name']
        self.saveName = tmap['saveName']
        self.inputLayer = []
        self.outputLayer = []
        self.inputLayerName = tmap['inputLayerName']
        self.outputLayerName = tmap['outputLayerName']

    @classmethod
    def to_yaml(cls, dumper, data):
        """
        Save this layer to yaml
        """
        return

    @classmethod
    def from_yaml(cls, loader, node):
        """
        Load this layer from yaml
        """
        return

class MoreIn(Layer):
    """
    Combine more than one input to form a output.
    The op supports combination only on one dimension/index.
    """
    LayerTypeName = 'MoreIn'
    yaml_tag = u'!MoreIn'

    def __init__(self):
        pass

    def __str__(self):
        return 'moreIn'

    def fillToObjMap(self):
        objDict = super(MoreIn, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(MoreIn, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(MoreIn.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = MoreIn()
        ret.loadFromObjMap(obj_dict)
        return ret
        

class MoreOut(Layer):
    """
    Connect one input to multiple output.
    """
    LayerTypeName = 'MoreOut'
    yaml_tag = u'!MoreOut'

    def __init__(self):
        pass

    def __str__(self):
        return 'moreIn'

    def fillToObjMap(self):
        objDict = super(MoreOut, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(MoreOut, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(MoreOut.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = MoreOut()
        ret.loadFromObjMap(obj_dict)
        return ret

class RawInput(Layer):
    """
    This is THE INPUT Class. Class type is checked during network building.
    """

    LayerTypeName = 'RawInput'
    yaml_tag = u'!RawInput'
    
    def __init__(self, inputsize):
        """
        Assume input size is (channel, column, row)
        """
        super(RawInput, self).__init__()
        self.size3 = inputsize
        self.size = None

    def __str__(self):
        ret = 'RawInput: {}'.format(self.size)
        return ret

    def setBatchSize(self, psize):
        """
        This method is suposed to called by network.setInput()
        """
        self.size = (psize, *self.size3)

    def forwardSize(self, inputsize):

        return [self.size]

    def fillToObjMap(self):
        objDict = super(RawInput, self).fillToObjMap()
        objDict['size'] = self.size
        return objDict

    def loadFromObjMap(self, tmap):
        super(RawInput, self).loadFromObjMap(tmap)
        self.size = tmap['size']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(RawInput.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = RawInput(obj_dict['size'][1:])
        ret.loadFromObjMap(obj_dict)
        return ret

        
class Conv2d(Layer):

    debugname = 'conv2d'
    LayerTypeName = 'Conv2d'
    yaml_tag = u'!Conv2d'
    
    def __init__(self, filter_size=(3,3),
                 input_feature=None, output_feature=None,
                 feature_map_multiplier=None,
                 subsample=(1,1), border='half'):
        """
        This 2d convolution deals with 4d tensor:
        (batch_size, feature map/channel, filter_row, filter_col)

        feature_map_multiplier always has a ligher priority
        than input_feature/output_feature
        """
        super(Conv2d, self).__init__()

        self.filterSize = filter_size
        self.inputFeature = input_feature
        self.outputFeature = output_feature
        self.mapMulti = feature_map_multiplier
        self.border = border
        self.subsample = subsample

        self.w = None
        
    def getpara(self):
        return [self.w]
    
    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        #print('conv2d.forward.type: {}'.format(inputimage.ndim))
        l3conv = T.nnet.conv2d(inputimage,
                               self.w,
                               border_mode=self.border,
                               subsample=self.subsample)
        return (l3conv, )
        
    def forwardSize(self, inputsize):
        # [size1, size2, size3], size: (32,1,28,28)
        # print("conv2d.size: {}, {}, {}".format(inputsize,self.mapMulti, self.inputFeature))
        isize = inputsize[0]

        if len(isize) != 4:
            raise IndexError
        if self.mapMulti is None and isize[1] != self.inputFeature:
            raise IndexError

        if self.mapMulti is not None:
            self.inputFeature = isize[1]
            self.outputFeature = int(self.inputFeature*self.mapMulti)

        weightIniter = winit.XavierInit()
        initweight = weightIniter.initialize((self.outputFeature,
                                              self.inputFeature,
                                              *self.filterSize))
        self.w = theano.shared(initweight, borrow=True)

        retSize = None
        if self.border == 'half':
            retSize = [(isize[0],
                        self.outputFeature,
                        int(isize[2]/self.subsample[0]),
                        int(isize[3]/self.subsample[1]))]
        else:
            raise NotImplementedError

        return retSize

    # The following methods are for saving and loading
    def fillToObjMap(self):
        objDict = super(Conv2d, self).fillToObjMap()
        objDict['filterSize'] = self.filterSize
        objDict['inputFeature'] = self.inputFeature
        objDict['outputFeature'] = self.outputFeature
        objDict['border'] = self.border
        objDict['subsample'] = self.subsample
        objDict['w'] = self.w

        return objDict

    def loadFromObjMap(self, tmap):
        super(Conv2d, self).loadFromObjMap(tmap)
        self.filterSize = tmap['filterSize']
        self.inputFeature = tmap['inputFeature']
        self.outputFeature = tmap['outputFeature']
        self.border = tmap['border']
        self.subsample = tmap['subsample']
        self.w = tmap['w']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Conv2d.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Conv2d(obj_dict['filterSize'], obj_dict['inputFeature'], obj_dict['outputFeature'],
                     None, obj_dict['subsample'], obj_dict['border'])
        ret.loadFromObjMap(obj_dict)
        return ret




class Pooling(Layer):
    debugname = 'pooling'
    LayerTypeName = 'Pooling'
    yaml_tag = u'!Pooling'
    
    def __init__(self, dsize=(2,2)):
        super(Pooling, self).__init__()
        self.size = dsize

    def getpara(self):
        return []

    def forward(self, inputtensor):
        inputactivation = inputtensor[0]
        return (pool_2d(inputactivation, self.size, ignore_border=True),)

    def forwardSize(self, inputsize):
        isize = inputsize[0]
        #print("pooling input size: {}".format(isize))

        if len(isize) != 4:
            raise IndexError

        return [(isize[0], isize[1], int(isize[2]/self.size[0]), int(isize[3]/self.size[1]))]

    def fillToObjMap(self):
        objDict = super(Pooling, self).fillToObjMap()
        objDict['size'] = self.size
        return objDict

    def loadFromObjMap(self, tmap):
        super(Pooling, self).loadFromObjMap(tmap)
        self.size = tmap['size']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Pooling.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Pooling(obj_dict['size'])
        ret.loadFromObjMap(obj_dict)
        return ret

class GlobalPooling(Layer):
    debugname = 'globalpooling'
    LayerTypeName = 'GlobalPooling'
    yaml_tag = u'!GlobalPooling'

    def __init__(self, pool_function=T.mean):
        super(GlobalPooling, self).__init__()

        self.poolFunc = pool_function

    def getpara(self):
        return []

    def forward(self, inputtensor):
        x = inputtensor[0]
        return [self.poolFunc(x.flatten(3), axis=2),]

    def forwardSize(self, inputsize):
        isize = inputsize[0]

        if len(isize) != 4:
            raise IndexError

        return [(isize[0], isize[1]),]

    def fillToObjMap(self):
        objDict = super(GlobalPooling, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(GlobalPooling, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(GlobalPooling.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = GlobalPooling()
        ret.loadFromObjMap(obj_dict)
        return ret


class FeaturePooling(Layer):
    """
    For maxout
    """
    def __init__(self, pool_size, axis=1, pool_function=theano.tensor.max):
        super(FeaturePooling, self).__init__()

        self.poolSize = pool_size
        self.axis = axis
        self.poolFunc = pool_function

    def getpara(self):
        return []

    def forward(self, inputtensor):
        x = inputtensor[0]

        inputShape = tuple(x.shape)
        poolShape = inputShape[:self.axis] + (inputShape[self.axis] // self.poolSize, self.poolSize) + inputShape[self.axis+1:]
        
        interData =T.reshape(x, poolShape)
        
        return [self.poolFunc(interData, axis=self.axis+1),]

    def forwardSize(self, inputsize):
        isize = list(inputsize[0])

        if len(isize) != 4:
            raise IndexError

        if isize[self.axis] % self.poolSize != 0:
            raise ValueError("input number of features is not multiple of the pool size.")

        outputSize = isize[:self.axis]
        outputSize += [isize[self.axis] // self.poolSize,]
        outputSize += isize[self.axis+1:]

        return [outputSize,]

    def fillToObjMap(self):
        objDict = super(FeaturePooling, self).fillToObjMap()
        objDict['poolSize'] = self.poolSize
        objDict['axis'] = self.axis
        objDict['poolFunc'] = 'max'
        return objDict

    def loadFromObjMap(self, tmap):
        super(FeaturePooling, self).loadFromObjMap(tmap)
        self.poolSize = objDict['poolSize']
        self.axis = objDict['axis']
        self.poolFunc = theano.tensor.max

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(FeaturePooling.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = FeaturePooling(obj_dict['poolSize'])
        ret.loadFromObjMap(obj_dict)
        return ret


class UpPooling(Layer):
    """
    This can be done as gradient/backward of pooling:

    The following code is from
    https://github.com/nanopony/keras-convautoencoder/blob/master/autoencoder_layers.py
    """
    def __init__(self):
        super(UpPooling, self).__init__()
        
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = K.repeat_elements(X, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = K.repeat_elements(X, self.size[0], axis=1)
            output = K.repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        
        f = T.grad(T.sum(self._pool2d_layer.get_output(train)), wrt=self._pool2d_layer.get_input(train)) * output

        return f

class Flatten(Layer):
    debugname = 'Flatten'
    LayerTypeName = 'Flatten'
    yaml_tag = u'!Flatten'
    
    def __init__(self):
        super(Flatten, self).__init__()

    def getpara(self):
        return []

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        return (T.flatten(inputimage, outdim=2),)

    def forwardSize(self, inputsize):
        isize = inputsize[0]

        if len(isize) != 4:
            raise IndexError

        return [(isize[0], isize[1]*isize[2]*isize[3], )]

    def fillToObjMap(self):
        objDict = super(Flatten, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(Flatten, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Flatten.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Flatten()
        ret.loadFromObjMap(obj_dict)
        return ret



class FullConn(Layer):

    debugname = 'Full Connection'
    LayerTypeName = 'FullConn'
    yaml_tag = u'!FullConn'
    
    def __init__(self, times=None, output=None, input_feature=None, output_feature=None):
        super(FullConn, self).__init__()
        if times is not None:
            self.times = times
        if output is not None:
            self.output = output

        weightIniter = winit.XavierInit()
        initweight = weightIniter.initialize((input_feature, output_feature))
        self.w = theano.shared(initweight, borrow=True)

        self.inputFeature = input_feature
        self.outputFeature = output_feature
        
        self.times = -1
        self.output = -1        

    def getpara(self):
        return (self.w, )

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        return (T.dot(inputimage, self.w), )

    def forwardSize(self, inputsize):

        #print(inputsize)
        #print(self.inputFeature)
        isize = inputsize[0]

        if len(isize) != 2:
            raise IndexError('Expect input dimension 2, get ' + str(len(isize)))
        if isize[1] != self.inputFeature:
            raise IndexError('Input size: ' +
                             str(isize[1]) +
                             ' is not equal to given input feature dim: ' +
                             str(self.inputFeature))

        return [(isize[0], self.outputFeature,)]

    def fillToObjMap(self):
        objDict = super(FullConn, self).fillToObjMap()
        objDict['inputFeature'] = self.inputFeature
        objDict['outputFeature'] = self.outputFeature
        objDict['w'] = self.w

        return objDict

    def loadFromObjMap(self, tmap):
        super(FullConn, self).loadFromObjMap(tmap)
        self.inputFeature = tmap['inputFeature']
        self.outputFeature = tmap['outputFeature']
        self.w = tmap['w']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(FullConn.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = FullConn(input_feature=obj_dict['inputFeature'],
                       output_feature=obj_dict['outputFeature'])
        ret.loadFromObjMap(obj_dict)
        return ret

class SoftMax(Layer):
    debugname = 'softmax'
    LayerTypeName = 'SoftMax'
    yaml_tag = u'!SoftMax'

    def __init__(self):
        super(SoftMax, self).__init__()
    
    def getpara(self):
        return []
    
    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        #e_x = T.exp(inputimage - inputimage.max(axis=1, keepdims=True))
        #out = e_x / e_x.sum(axis=1, keepdims=True)
        #return (T.nnet.softmax(inputimage),)
        e_x = T.exp(inputimage - inputimage.max(axis=1).dimshuffle(0, 'x'))
        return (e_x / e_x.sum(axis=1).dimshuffle(0, 'x'),)

    def forwardSize(self, inputsize):

        return inputsize

    def fillToObjMap(self):
        objDict = super(SoftMax, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(SoftMax, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(SoftMax.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = SoftMax()
        ret.loadFromObjMap(obj_dict)
        return ret

class BatchNormalization(Layer):
    
    debugname = 'bn'
    LayerTypeName = 'BatchNormalization'
    yaml_tag = u'!BatchNormalization'

    def __init__(self):
        super(BatchNormalization, self).__init__()

        self.gamma = None
        self.beta = None
        self.meanStats = None
        self.varStats = None

        self.statsRate = 0.9

    def getpara(self):
        return [self.gamma, self.beta]

    def getExtraPara(self, inputtensor):
        x = inputtensor[0]
        return [(self.meanStats, self.meanStats*self.statsRate + x.mean(0)*(1-self.statsRate))
                , (self.varStats, self.varStats*self.statsRate + x.var(0)*(1-self.statsRate))]

    def forward(self, inputtensor):
        x = inputtensor[0]
        #out = T.nnet.bn.batch_normalization(x, self.gamma, self.beta, x.mean(axis=0), x.std(axis=0), mode='high_mem')
        xmean = x.mean(axis=0)
        xvar = x.var(axis=0)
        tx = (x - xmean) / T.sqrt(xvar+0.001)
        out = tx*self.gamma + self.beta
        return (out,)

    def predictForward(self, inputtensor):
        x = inputtensor[0]
        #out = T.nnet.bn.batch_normalization(x, self.gamma, self.beta, self.meanStats, self.stdStats, mode='high_mem')
        tx = (x - self.meanStats) / T.sqrt(self.varStats+0.001)
        out = tx*self.gamma + self.beta
        return (out,)

    def forwardSize(self, inputsize):
        #print(inputsize)
        xsize = inputsize[0]
        isize = xsize[1:]
        #print('bn.size: {}'.format(isize))
        
        betaInit = floatX(np.zeros(isize))
        self.beta = theano.shared(betaInit, name=self.name+'beta', borrow=True)

        gammaInit = floatX(np.ones(isize))
        self.gamma = theano.shared(gammaInit, name=self.name+'gamma', borrow=True)

        meanInit = floatX(np.zeros(isize))
        self.meanStats = theano.shared(meanInit, borrow=True)

        varInit = floatX(np.ones(isize))
        self.varStats = theano.shared(varInit, borrow=True)

        return inputsize

    # The following methods are for saving and loading
    def fillToObjMap(self):
        objDict = super(BatchNormalization, self).fillToObjMap()
        objDict['gamma'] = self.gamma
        objDict['beta'] = self.beta
        objDict['meanStats'] = self.meanStats
        objDict['varStats'] = self.varStats

        return objDict

    def loadFromObjMap(self, tmap):
        super(BatchNormalization, self).loadFromObjMap(tmap)
        self.gamma = tmap['gamma']
        self.beta = tmap['beta']
        self.meanStats = tmap['meanStats']
        self.varStats = tmap['varStats']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(BatchNormalization.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = BatchNormalization()
        ret.loadFromObjMap(obj_dict)
        return ret
        
import theano
import theano.tensor as T
import mlbase.networkhelper as N
import numpy as np

class NonLinear(N.Layer):
    LayerTypeName = 'NonLinear'
    yaml_tag = u'!NonLinear'
    
    def __str__(self):
        return ret

    def fillToObjMap(self):
        objDict = super(NonLinear, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(NonLinear, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(NonLinear.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = NonLinear()
        ret.loadFromObjMap(obj_dict)
        return ret

class Relu(NonLinear):
    debugname = 'relu'
    LayerTypeName = 'Relu'
    yaml_tag = u'!Relu'

    def __init__(self):
        super(Relu, self).__init__()

    def getpara(self):
        return []

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        return (T.nnet.relu(inputimage),)

    def forwardSize(self, inputsize):
        return inputsize

    def __str__(self):
        ret = 'Relu:'
        return ret

    def fillToObjMap(self):
        objDict = super(Relu, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(Relu, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Relu.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Relu()
        ret.loadFromObjMap(obj_dict)
        return ret

class Sine(NonLinear):
    debugname = 'sine'
    LayerTypeName = 'Sine'
    yaml_tag = u'!Sine'

    def __init__(self):
        super(Sine, self).__init__()
    
    def getpara(self):
        return []

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        return (T.sin(inputimage),)

    def fillToObjMap(self):
        objDict = super(Sine, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(Sine, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Sine.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Sine()
        ret.loadFromObjMap(obj_dict)
        return ret

class Cosine(NonLinear):
    debugname = 'cos'
    LayerTypeName = 'Cosine'
    yaml_tag = u'!Cosine'

    def __init__(self):
        super(Cosine, self).__init__()
    
    def getpara(self):
        return []

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        return (T.cos(inputimage*10),)

    def fillToObjMap(self):
        objDict = super(Cosine, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(Cosine, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Cosine.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Cosine()
        ret.loadFromObjMap(obj_dict)
        return ret

class AbsoluteValue(NonLinear):
    debugname = 'absolutevalue'
    LayerTypeName = 'AbsoluteValue'
    yaml_tag = u'!AbsoluteValue'

    def __init__(self):
        super(AbsoluteValue, self).__init__()
    
    def getpara(self):
        return []

    def forward(self, inputtensor):
        X = inputtensor[0]
        ret = abs(X)
        return (ret,)

    def fillToObjMap(self):
        objDict = super(AbsoluteValue, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(AbsoluteValue, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(AbsoluteValue.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = AbsoluteValue()
        ret.loadFromObjMap(obj_dict)
        return ret


class Triangle(NonLinear):
    debugname = 'triangle'
    LayerTypeName = 'Triangle'
    yaml_tag = u'!Triangle'

    def __init__(self):
        super(Triangle, self).__init__()
    
    def getpara(self):
        return []

    def forward(self, inputtensor):
        X = inputtensor[0]
        ret = abs((X/2 - T.floor(X/2))*2-1)*2-1
        return (ret,)

    def fillToObjMap(self):
        objDict = super(Triangle, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(Triangle, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Triangle.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Triangle()
        ret.loadFromObjMap(obj_dict)
        return ret

# every layer has a different 'a' that can be learned
class Sine2(NonLinear):
    debugname = 'sine'
    LayerTypeName = 'Sine'
    yaml_tag = u'!Sine'

    def __init__(self):
        super(Sine, self).__init__()
        self.a = theano.shared(1.0)
    
    def getpara(self):
        return [self.a]

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        return (T.sin(self.a*inputimage),)

    def fillToObjMap(self):
        objDict = super(Sine, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(Sine, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Sine.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Sine()
        ret.loadFromObjMap(obj_dict)
        return ret

class Cosine2(NonLinear):
    debugname = 'cos'
    LayerTypeName = 'Cosine'
    yaml_tag = u'!Cosine'

    def __init__(self):
        super(Cosine, self).__init__()
        self.a = theano.shared(1.0)
    
    def getpara(self):
        return [self.a]

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        return (T.cos(self.a*inputimage),)

    def fillToObjMap(self):
        objDict = super(Cosine, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(Cosine, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Cosine.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Cosine()
        ret.loadFromObjMap(obj_dict)
        return ret

# every feature map has a different 'a' that can be learned
class Sine3(NonLinear):
    debugname = 'sine'
    LayerTypeName = 'Sine'
    yaml_tag = u'!Sine'

    def __init__(self):
        super(Sine, self).__init__()
    
    def getpara(self):
        return [self.a]

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        if len(inputimage.ndim) == 4:
            return (T.cos(self.a.dimshuffle('x', 0, 'x', 'x')*inputimage),)
        if len(inputimage.ndim) == 2:
            return (T.cos(self.a*inputimage),)
        else:
            raise NotImplementedError
    
    def forwardSize(self, inputsize):
        isize = inputsize[0]
        ainit = np.zeros((isize[1],))
        self.a = theano.shared(ainit, borrow=True)
        if len(isize) == 4:
            return [(isize[0], isize[1], isize[2], isize[3])]
        if len(isize) == 2:
            return [(isize[0], isize[1])]
        else:
            raise IndexError

    def fillToObjMap(self):
        objDict = super(Sine, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(Sine, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Sine.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Sine()
        ret.loadFromObjMap(obj_dict)
        return ret

class Cosine3(NonLinear):
    debugname = 'cos'
    LayerTypeName = 'Cosine'
    yaml_tag = u'!Cosine'

    def __init__(self):
        super(Cosine, self).__init__()
    
    def getpara(self):
        return [self.a]

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        if len(inputimage.ndim) == 4:
            return (T.cos(self.a.dimshuffle('x', 0, 'x', 'x')*inputimage),)
        if len(inputimage.ndim) == 2:
            return (T.cos(self.a*inputimage),)
        else:
            raise NotImplementedError
    
    def forwardSize(self, inputsize):
        isize = inputsize[0]
        ainit = np.zeros((isize[1],))
        self.a = theano.shared(ainit, borrow=True)
        if len(isize) == 4:
            return [(isize[0], isize[1], isize[2], isize[3])]
        if len(isize) == 2:
            return [(isize[0], isize[1])]
        else:
            raise IndexError

    def fillToObjMap(self):
        objDict = super(Cosine, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(Cosine, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Cosine.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Cosine()
        ret.loadFromObjMap(obj_dict)
        return ret

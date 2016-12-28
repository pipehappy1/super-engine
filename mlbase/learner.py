import numpy as np

class Learner:
    def reset(self):
        """
        Reinitialize model parameter.
        """
        pass

class SupervisedLearner(Learner):
    def train(self, X, Y):
        pass

    def predict(self, X):
        pass

class UnsupervisedLearner(Learner):
    def train(self, X):
        pass

    def predict(self, X):
        pass


class CrossValidation:
    def __init__(self, fold=10, learner=None,
                 train_input=None, train_output=None,
                 test_input=None, test_output=None,
                 validate_input=None, validate_output=None):
        self.fold = fold
        self.learner = learner
        self.trainData = train_input
        self.trainLabel = train_output
        self.testData = test_input
        self.testLabel = test_output
        if validate_input is not None or validate_output is not None:
            raise NotImplementedError('cross validation do not support extra validation set.')

        self.foldIndex = CrossValidation.splitData(self.trainLabel, self.fold)

    @staticmethod
    def splitData(trainLabel, fold):
        numClass = trainLabel.shape[1]
        sampleIndex = np.arange(len(trainLabel))

        retFolds = {}
        for classIndex in range(numClass):
            retFolds[classIndex] = {}
            
            classSampleIndex = sampleIndex[trainLabel[:, classIndex]==1]
            classFoldSize = (len(classSampleIndex) // fold) + 1
            
            for foldIndex in range(fold):
                retFolds[classIndex][foldIndex] = classSampleIndex[foldIndex*classFoldSize :
                                                                   min((foldIndex+1)*classFoldSize, len(classSampleIndex)+1)]

        return retFolds
            
    @staticmethod
    def getNFold(classFoldIndex, nFold, fold):
        numClass = len(classFoldIndex)

        trainIndex = None
        validateIndex = None
        for foldIndex in range(fold):
            for classIndex in range(numClass):
                if foldIndex == nFold:
                    if validateIndex is None:
                        validateIndex = classFoldIndex[classIndex][foldIndex]
                    else:
                        validateIndex = np.hstack((validateIndex, classFoldIndex[classIndex][foldIndex]))
                else:
                    if trainIndex is None:
                        trainIndex = classFoldIndex[classIndex][foldIndex]
                    else:
                        trainIndex = np.hstack((trainIndex, classFoldIndex[classIndex][foldIndex]))

        np.random.shuffle(trainIndex)
        np.random.shuffle(validateIndex)
        return trainIndex, validateIndex
                        

    def train(self):
        # assume the data is shuffled.
        models = []
        
        for i in range(self.fold):
            print("fold: {}".format(i))
            self.learner.reset()
            
            trainIndex, validateIndex = CrossValidation.getNFold(self.foldIndex, i, self.fold)
            print(trainIndex)
            print(validateIndex)
            print(self.learner.learner)
            thisTrainData = self.trainData[trainIndex, :]
            thisTrainLabel = self.trainLabel[trainIndex, :]
            thisValidateData = self.trainData[validateIndex, :]
            thisValidateLabel = self.trainLabel[validateIndex, :]

            for epochIndex in range(100):
                self.learner.train(thisTrainData, thisTrainLabel)
                testError = 1 - np.mean(np.argmax(self.testLabel, axis=1) == np.argmax(self.learner.predict(self.testData), axis=1))
                validateError = 1 - np.mean(np.argmax(thisValidateLabel, axis=1) == np.argmax(self.learner.predict(thisValidateData), axis=1))
                print("epochIndex: {}, testError: {}, validateError: {}".format(epochIndex, testError, validateError))


            
if __name__ == '__main__':
    import mlbase.loaddata as l
    import mlbase.networkhelper as N
    import mlbase.activation as act
    
    trX, trY, teX, teY = l.load_mnist()

    network = N.Network()
    network.modelPrefix = 'cvTest'

    network.setInput(N.RawInput((1, 28,28)))
    network.append(N.Conv2d(feature_map_multiplier=32))
    network.append(act.Relu())
    network.append(N.Pooling())
    network.append(N.Conv2d(feature_map_multiplier=2))
    network.append(act.Relu())
    network.append(N.Pooling())
    network.append(N.Conv2d(feature_map_multiplier=2))
    network.append(act.Relu())
    network.append(N.Pooling())
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(act.Relu())
    network.append(N.FullConn(input_feature=1152*2, output_feature=10))
    network.append(N.SoftMax())

    network.build()

    cv = CrossValidation(fold=10, learner=network, train_input=trX, train_output=trY, test_input=teX, test_output=teY)
    cv.train()

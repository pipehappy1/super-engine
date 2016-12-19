class Learner:
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

class TensorData:
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

    def train(self):
        # assume the data is shuffled.
        pass
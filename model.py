
class BaseModel(object):
    """Base class for our model"""

    def get_predictions(self):
        raise Error('Not implemented')

    def get_loss(self):
        raise Error('Not implemented')

    def get_training_operation(self):
        raise Error('Not implemented')

    def get_global_step(self):
        raise Error('Not implemented')

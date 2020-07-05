from abc import ABC
from threading import Thread, Event, Lock
import svm
import time 

class Model(ABC):
    FACTORY = {}
    @classmethod
    def factory(cls,model_name):
        return cls.FACTORY[model_name]()
    def __init__(self):
        ''' init should load the model '''
        pass
    def classify(self,image):
        pass

class SVMModel(Model):
    def __init__(self):
        self.svm = svm.get_saved_svm()
    def classify(self,image):
        return svm.get_prediction(self.svm, image)

Model.FACTORY["svm"] = SVMModel
Model.FACTORY["svm2"] = SVMModel

class InferenceServer:
    """
        abstracts model loading and classification operations
    """
    def __init__(self):
        self.model = None
        self.model_lock = Lock()
        self.load_event = Event()
        self.load_event.set()

        self.load_model_name = "svm" # default model

        # load_worker waits in the background and does model loading when requested
        Thread(target=self.load_worker,daemon=True).start()

    def load_worker(self):
        '''
            background thread that loads and unloads models
        '''
        while True:
            # wait for the load request
            self.load_event.wait()

            # Load the model into a temp variable
            tic = time.time()
            model_temp = Model.factory(self.load_model_name)
            print("{} took {:.4f} seconds to load".format(self.load_model_name,time.time()-tic))
            self.load_model_name = None

            with self.model_lock:
                self.model = model_temp

            self.load_event.clear()
    
    def request_model_change(self, model_name):
        if model_name not in Model.FACTORY:
            raise Exception("{} not in model factory".format(model_name))
            return
        self.load_model_name = model_name
        self.load_event.set()

    def classify(self, image):
        result = None
        if self.model:
            with self.model_lock:
                tic = time.time()
                result = self.model.classify(image)
                print("{:.4f} seconds to classify".format(time.time()-tic))
        return result 
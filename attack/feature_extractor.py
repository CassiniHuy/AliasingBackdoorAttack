from operator import attrgetter

class FeatureExtractor():
    def __init__(self, model, return_layer: str = None):
        """Get the intermediate outputs from model

        Args:
            model (nn.Module): the model
            return_layer (str): layer named(returned by model.named_modules)
        """        
        self.model = model
        self.return_layer = return_layer
        self._handle = None
        self._feats = None
        self.hook()
    
    def hook(self):
        if self.return_layer is None:
            return
        layer = attrgetter(self.return_layer)(self.model)
        
        def hook(module, input, output):
            self._feats = output
            raise StopIteration()
        try:
            h = layer.register_forward_hook(hook)
        except AttributeError as e:
            raise AttributeError(f'Module {self.return_layer} not found')
        
        self._handle = h
            
    def unhook(self):
        if self._handle is None:
            return
        self._handle.remove()
        self._handles = None

    def rehook(self):
        self.unhook()
        self.hook()
        
    def __call__(self, *args, **kwargs):
        self._feats = None
        try:
            self._feats = self.model(*args, **kwargs)
        except StopIteration:
            pass
        return self._feats
    
    @staticmethod
    def extract_feature(x, model, feats_layer: str):
        extractor = FeatureExtractor(model, feats_layer)
        y = extractor(x)
        extractor.unhook()
        return y

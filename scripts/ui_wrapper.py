import os

class UIWrapper:
    def title(self) -> str:
        raise NotImplementedError
    
    def show(self) -> None:
        raise NotImplementedError
    
    def setup_ui(self) -> list:
        raise NotImplementedError
    
    def before_process(self, p, *args, **kwargs):
        raise NotImplementedError

    def process(self, p, *args, **kwargs):
        raise NotImplementedError

    def before_process_batch(self, p, *args, **kwargs):
        raise NotImplementedError

    def process_batch(self, p, *args, **kwargs):
        raise NotImplementedError

    def postprocess_batch(self, p, *args, **kwargs):
        raise NotImplementedError

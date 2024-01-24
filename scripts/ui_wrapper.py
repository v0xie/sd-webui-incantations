import os

class UIWrapper:
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []

    def title(self) -> str:
        raise NotImplementedError
    
    def setup_ui(self, is_img2img) -> list:
        raise NotImplementedError

    def get_infotext_fields(self) -> list:
        return self.infotext_fields

    def get_paste_field_names(self) -> list:
        return self.paste_field_names
    
    def before_process(self, p, *args, **kwargs):
        pass

    def process(self, p, *args, **kwargs):
        pass

    def before_process_batch(self, p, *args, **kwargs):
        pass

    def process_batch(self, p, *args, **kwargs):
        pass

    def postprocess_batch(self, p, *args, **kwargs):
        pass
    
    def unhook_callbacks(self) -> None:
        pass

    def get_xyz_axis_options(self) -> dict:
        raise NotImplementedError

def arg(p, field_name: str, variable_name:str, default=None, **kwargs):
        """ Get argument from field_name or variable_name, or default if not found """
        return getattr(p, field_name, kwargs.get(variable_name, None))

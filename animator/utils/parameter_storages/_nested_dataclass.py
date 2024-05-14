from dataclasses import dataclass, is_dataclass

def nested_dataclass(*args, **kwargs):
    def wrapper(in_cls):
        in_cls = dataclass(in_cls, **kwargs)
        original_init = in_cls.__init__
        def __init__(self, *args, **kwargs):
            for name, val in kwargs.items():
                field_type = in_cls.__annotations__.get(name, None)
                if is_dataclass(field_type) and isinstance(val, dict):
                    kwargs[name] = field_type(**val)
            original_init(self, *args, **kwargs)
        in_cls.__init__ = __init__
        return in_cls
    return wrapper(args[0]) if args else wrapper
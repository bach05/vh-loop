# script/core/registry.py

_MODEL_ADAPTERS = {}
_TRAINING_BACKENDS = {}


def register_model_adapter(name):
    def decorator(cls):
        _MODEL_ADAPTERS[name] = cls
        return cls
    return decorator


def get_model_adapter(name, **kwargs):
    if name not in _MODEL_ADAPTERS:
        raise ValueError(f"Unknown model adapter: {name}")
    return _MODEL_ADAPTERS[name](**kwargs)


def register_training_backend(name):
    def decorator(cls):
        _TRAINING_BACKENDS[name] = cls
        return cls
    return decorator


def get_training_backend(name, **kwargs):
    if name not in _TRAINING_BACKENDS:
        raise ValueError(f"Unknown training backend: {name}")
    return _TRAINING_BACKENDS[name](**kwargs)
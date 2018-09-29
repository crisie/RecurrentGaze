# !/usr/bin/env python
# title           :model/__init__.py
# description     :model getter
# author          :Cristina Palmero
# date            :30092018
# version         :1.0
# usage           : -
# notes           : -
# python_version  :3.5.5
# ==============================================================================
from importlib import import_module
from .model import Model


def get_model(model_name: str, *args, **kwargs):
    """
    Get model given name. Raise Error if given name does not exist.
    :param model_name: model name
    :param args:
    :param kwargs:
    :return: model
    """
    try:
        if '.' in model_name:
            module_name, class_name = model_name.rsplit('.', 1)
        else:
            module_name = model_name
            class_name = model_name.capitalize().replace("_","")

        model_module = import_module('.' + module_name, package='models')

        model_class = getattr(model_module, class_name)

        instance = model_class(*args, **kwargs)

    except (AttributeError, ModuleNotFoundError):
        raise ImportError('{} is not part of our model/architecture collection.'.format(model_name))
    else:
        if not issubclass(model_class, Model):
            raise ImportError("{} is not a valid model/architecture.".format(model_class))

    return instance

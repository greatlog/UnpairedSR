import importlib
import logging
import os
import os.path as osp

from utils.registry import MODEL_REGISTRY

logger = logging.getLogger("base")

model_folder = osp.dirname(__file__)
model_names = [
    osp.splitext(osp.basename(v))[0]
    for v in os.listdir(model_folder)
    if v.endswith("_model.py")
]
_model_modules = [
    importlib.import_module(f"models.{file_name}") for file_name in model_names
]


def create_model(opt, **kwarg):
    model = opt["model"]
    m = MODEL_REGISTRY.get(model)(opt, **kwarg)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m

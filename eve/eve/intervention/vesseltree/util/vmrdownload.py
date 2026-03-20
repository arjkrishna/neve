import os
import logging
import sys
import requests
import shutil
import wget

logger = logging.getLogger("VMR_Download")


def download_vmr_files(model: str) -> str:
    path_util = os.path.dirname(os.path.realpath(__file__))
    path_vessel_tree, _ = os.path.split(path_util)
    path_intervention, _ = os.path.split(path_vessel_tree)
    path_eve, _ = os.path.split(path_intervention)
    path_lib_base, _ = os.path.split(path_eve)
    path_data = os.path.join(path_lib_base, ".data")
    path_vmr = os.path.join(path_data, "vmr")
    path_model = os.path.join(path_vmr, model)

    if not os.path.exists(path_data):
        os.mkdir(path_data)
    if not os.path.exists(path_vmr):
        os.mkdir(path_vmr)

    logger.info(
        "Downloading vascular models from https://vascularmodel.com/. Please cite appropriately when using for publications."
    )

    info_log = f"Downloading {model} to {path_model}"
    logger.info(info_log)

    model_zip_url = f"https://www.vascularmodel.com/svprojects/{model}.zip"
    model_zip_path = os.path.join(path_vmr, f"{model}.zip")
    if not os.path.exists(model_zip_path):
        _download(model_zip_url, model_zip_path)

    shutil.unpack_archive(model_zip_path, path_vmr)
    path_model_mesh = os.path.join(path_model, "Meshes")

    for filename in os.listdir(path_model_mesh):
        old_path = os.path.join(path_model_mesh, filename)
        _, extension = os.path.splitext(filename)
        new_path = os.path.join(path_model_mesh, f"{model}{extension}")
        if not os.path.isfile(new_path):
            info_log = f"Creating {model}{extension} in {path_model_mesh}"
            logger.info(info_log)
            shutil.copyfile(old_path, new_path)
        else:
            debug_log = f"{model}{extension} already exists in {path_model_mesh}."
            logger.debug(debug_log)

    return path_model


def _download(url, local_path):
    wget.download(url, local_path)
    
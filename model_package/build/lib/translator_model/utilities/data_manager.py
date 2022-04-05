import csv
import pickle
import warnings
from typing import Any, List, Optional, TextIO, Tuple, Union

from translator_model import __version__ as _version
from translator_model.config.base import (
    DATASET_DIR,
    TRAINED_PROC_DIR,
    WEIGHTS_DIR,
    config,
)
from translator_model.pipeline import Pipeline
from translator_model.utilities.model_tools import arabic_normalizer, english_normalizer


def func():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    func()
    import tensorflow as tf
    from tensorflow.keras.layers import TextVectorization


def load_dataset(
    *,
    file_name: Union[str, TextIO],
    file_type: str,
    pairs: bool = True,
    delimiter: str = "\t",
) -> Tuple[List[Any], Optional[List[Any]]]:
    #
    print("Attempting to load data...")
    loop = True
    while loop:
        try:
            ["pkl", "txt", "csv"].index(file_type)
            if file_type == "pkl":
                with open(f"{DATASET_DIR}/{file_name}", "rb") as f:
                    data = pickle.load(f)
                    if pairs:
                        inp, targ = [inp for inp, _ in data], [targ for _, targ in data]
                    else:
                        inp, targ = data, None
                    f.close()

            # should come as comma seperate pairs, line separated txt or csv documents
            else:
                with open(
                    f"{DATASET_DIR}/{file_name}",
                    "r",
                    newline="\n",
                    encoding="utf8",
                ) as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    if pairs:
                        data = [
                            ["".join(line[i]) for i in range(len(line))]
                            for line in reader
                        ]
                        inp, targ = [inp for inp, _ in data], [targ for _, targ in data]
                    else:
                        data = ["".join(line) for line in reader]
                        inp, targ = data, None
                    f.close
            print("Data loaded successfully.")
            loop = False
        except ValueError:
            print(
                "Not a valid file type; only 'pkl', 'txt', or 'csv' filetypes supported."
            )
            break

    return inp, targ


def remove_old_processor(*, files_to_keep: List[str]) -> None:
    """
    Iterates through every file in the target directory and removes all but the
    new pipeline file and the __init__.py file.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_PROC_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def save_processor(*, processor_to_persist: TextVectorization, inp: bool) -> None:

    if inp is True:
        save_file_name = f"{config.app_config.inp_processor_save_file}{_version}.pkl"

    else:
        save_file_name = f"{config.app_config.targ_processor_save_file}{_version}.pkl"

    save_path = TRAINED_PROC_DIR / save_file_name
    remove_old_processor(
        files_to_keep=[
            f"{config.app_config.inp_processor_save_file}{_version}.pkl",
            f"{config.app_config.targ_processor_save_file}{_version}.pkl",
        ]
    )

    pickle.dump(
        {
            "config": processor_to_persist.get_config(),
            "weights": processor_to_persist.get_weights(),
        },
        open(save_path, "wb"),
    )


def load_processor(
    *, file_name: str, normalizer: str
) -> tf.keras.layers.TextVectorization:

    file_path = TRAINED_PROC_DIR / file_name
    with open(file_path, "rb") as f:
        loaded_processor = pickle.load(f)
        f.close()
    if normalizer == "english":
        _normalizer = english_normalizer
    elif normalizer == "arabic":
        _normalizer = arabic_normalizer
    else:
        print("Normalizer not recognized. Enter either 'english' or 'arabic'.")

    processor = tf.keras.layers.TextVectorization(
        max_tokens=loaded_processor["config"]["max_tokens"],
        output_mode="int",
        output_sequence_length=loaded_processor["config"]["output_sequence_length"],
        standardize=_normalizer,
    )

    processor.set_weights(loaded_processor["weights"])

    return processor


def remove_old_weights(*, files_to_keep: List[str]) -> None:
    """
    Iterates through every file in the target directory and removes all but the
    new pipeline file and the __init__.py file.

    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in WEIGHTS_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def save_weights(*, translator_to_persist: Pipeline) -> None:
    # define name pipeline of newely trained model
    save_file_name = f"{config.app_config.weights_save_file}{_version}"
    save_path = WEIGHTS_DIR / save_file_name

    remove_old_weights(
        files_to_keep=[
            save_file_name + ".data-00000-of-00001",
            save_file_name + ".index",
        ]
    )
    translator_to_persist.save_weights(save_path, save_format="tf")


def load_pipeline(
    *, weights_file_name: str = f"{config.app_config.weights_save_file}{_version}"
) -> Pipeline:

    inp_processor = load_processor(
        file_name=f"{config.app_config.inp_processor_save_file}{_version}.pkl",
        normalizer="english",
    )
    targ_processor = load_processor(
        file_name=f"{config.app_config.targ_processor_save_file}{_version}.pkl",
        normalizer="arabic",
    )

    file_path = WEIGHTS_DIR / weights_file_name
    pipeline = Pipeline(
        weights_file_name=file_path,
        trained_inp_processor=inp_processor,
        trained_targ_processor=targ_processor,
    )

    return pipeline

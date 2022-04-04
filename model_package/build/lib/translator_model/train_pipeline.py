import logging

from translator_model.config.base import config
from translator_model.pipeline import Pipeline
from translator_model.utilities.data_manager import (
    load_dataset,
    save_processor,
    save_weights,
)

logging.getLogger("tensorflow").setLevel(logging.INFO)


def train() -> None:
    inp, targ = load_dataset(
        file_name=config.app_config.train_data_file,
        file_type=config.app_config.train_data_file_type,
        pairs=True,
        delimiter="\t",
    )

    # train preprocessors and RNN
    pipeline = Pipeline(inp_data=inp, targ_data=targ)
    pipeline.train_preprocessors()
    pipeline.train_translator()

    # persist pipeline
    for p, boolean in zip(
        [pipeline.input_text_processor, pipeline.output_text_processor],
        [True, False],
    ):
        save_processor(processor_to_persist=p, inp=boolean)
    save_weights(translator_to_persist=pipeline.translator)


if __name__ == "__main__":
    # print("Attempting to train...")
    train()

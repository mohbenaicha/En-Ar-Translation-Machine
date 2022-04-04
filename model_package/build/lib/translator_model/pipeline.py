import warnings
from typing import List, Optional

from translator_model.config.base import config
from translator_model.utilities.model_tools import (
    BatchLogs,
    MaskedLoss,
    TrainTranslator,
    arabic_normalizer,
    english_normalizer,
    load_model,
)


def func():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    func()
    import tensorflow as tf
    from tensorflow.keras.layers import TextVectorization


class Pipeline:
    def __init__(
        self,
        inp_data: List[str] = None,
        targ_data: List[str] = None,
        weights_file_name: str = None,
        trained_inp_processor: Optional[TextVectorization] = None,
        trained_targ_processor: Optional[TextVectorization] = None,
    ):

        self.inp_data = inp_data
        self.targ_data = targ_data
        self.weights_file_name = weights_file_name

        if weights_file_name:
            assert trained_inp_processor and trained_targ_processor
            self.weights_file_name = weights_file_name
        self.input_text_processor = trained_inp_processor
        self.output_text_processor = trained_targ_processor

    def train_preprocessors(
        self,
        inp_data: List[str] = None,
        targ_data: List[str] = None,
        max_vocab_size: int = config.model_config.max_vocab_size,
    ) -> None:

        self.max_vocab_size = max_vocab_size
        self.arabic_normalizer = arabic_normalizer
        self.english_normalizer = english_normalizer

        self.input_text_processor = tf.keras.layers.TextVectorization(
            standardize=self.english_normalizer, max_tokens=self.max_vocab_size
        )

        self.output_text_processor = tf.keras.layers.TextVectorization(
            standardize=self.arabic_normalizer, max_tokens=self.max_vocab_size
        )

        if inp_data:
            assert all(
                [
                    isinstance(data, list) and all([isinstance(i, str) for i in data])
                    for data in [inp_data, targ_data]
                ]
            )
        if not any([self.inp_data, self.targ_data]):
            self.inp_data, self.targ_data = inp_data, targ_data

        if not all([self.inp_data, self.targ_data]):
            print(
                """Error, no data  specified at pipeline instantiation
                    or when calling train methods. You must enter
                    arguments for inp_data and targ_data arguments."""
            )
        else:
            print("Fitting input preprocessor...")
            self.input_text_processor.adapt(self.inp_data)
            print("Fitting target preprocessor...")
            self.output_text_processor.adapt(self.targ_data)

    def train_translator(
        self,
        inp_data: List[str] = None,
        targ_data: List[str] = None,
        n_units: int = config.model_config.units,
        n_dims: int = config.model_config.embedding_dim,
        n_epochs: int = config.model_config.epochs,
    ) -> TrainTranslator:

        if inp_data:
            assert all(
                [
                    isinstance(data, list) and all([isinstance(i, str) for i in data])
                    for data in [inp_data, targ_data]
                ]
            )
        if not any([self.inp_data, self.targ_data]):
            print("self.inp_data, self.targ_data do not exist")
            self.inp_data, self.targ_data = inp_data, targ_data

        if not all([self.inp_data, self.targ_data]):
            print(
                """Error, no data  specified at pipeline instantiation
                    or when calling train methods. You must enter
                    arguments for inp_data and targ_data arguments."""
            )

        if not self.weights_file_name:

            self.translator = TrainTranslator(
                embedding_dim=n_dims,
                units=n_units,
                input_text_processor=self.input_text_processor,
                output_text_processor=self.output_text_processor,
                use_tf_function=True,
            )

            self.translator.compile(optimizer=tf.optimizers.Adam(), loss=MaskedLoss())

            self.batch_loss = BatchLogs("batch_loss")
            if self.inp_data:
                self.dataset = tf.data.Dataset.from_tensor_slices(
                    (self.inp_data, self.targ_data)
                ).shuffle(len(self.inp_data))
                self.dataset = self.dataset.batch(config.model_config.batch_size)

                print("Fitting translator ...")
                self.translator.fit(
                    self.dataset,
                    epochs=n_epochs,
                    callbacks=[self.batch_loss],
                    verbose=1,
                )

        else:
            print(
                """You are calling a train method while specifying a weight
                    file. To translate using a trained translator, do not call
                    the train method. This pipeline class is not made to handle
                    continued traning of pretrained pipelines."""
            )

    def translate(self, query: List[str]) -> List[str]:

        translator = load_model(
            input_processor=self.input_text_processor,
            output_processor=self.output_text_processor,
            weights=self.weights_file_name,
            embedding_dim=config.model_config.embedding_dim,
            n_units=config.model_config.units,
        )

        query = tf.constant(query)
        result = translator.translate_unrolled(input_text=query)
        return [result["text"][i].numpy().decode() for i in range(len(result["text"]))]

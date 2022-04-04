import typing
import warnings
from typing import Any, Tuple

import numpy as np


def func():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    func()
    import tensorflow as tf
    import tensorflow_text as tf_text
    from tensorflow.keras.layers import TextVectorization


def english_normalizer(text):
    text = tf_text.normalize_utf8(text, "NFKD")
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-z]", " ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[START]", text, "[END]"], separator=" ")
    return tf.convert_to_tensor(text)


def arabic_normalizer(text):
    text = tf_text.normalize_utf8(text, "NFKD")
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ ء-ي]", "")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[START]", text, "[END]"], separator=" ")
    return tf.convert_to_tensor(text)


class ShapeChecker:
    def __init__(self):
        # Keep a cache of every axis-name seen
        self.shapes = {}

    def __call__(self, tensor, names=("name", "name", "name"), broadcast=False):
        if not tf.executing_eagerly():
            return

        if isinstance(names, str):
            names = (names,)

        shape = tf.shape(tensor)
        rank = tf.rank(tensor)

        if rank != len(names):
            raise ValueError(
                f"Rank mismatch:\n"
                f"    found {rank}: {shape.numpy()}\n"
                f"    expected {len(names)}: {names}\n"
            )

        for i, name in enumerate(names):
            if isinstance(name, int):
                old_dim = name
            else:
                old_dim = self.shapes.get(name, None)
            new_dim = shape[i]

            if broadcast and new_dim == 1:
                continue

            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(
                    f"Shape mismatch for dimension: '{name}'\n"
                    f"    found: {new_dim}\n"
                    f"    expected: {old_dim}\n"
                )


class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.input_vocab_size = input_vocab_size
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            # Return the sequence and state
            return_sequences=True,
            return_state=True,
            recurrent_initializer="he_normal",
        )

    def call(self, tokens, state=None):
        vectors = self.embedding(tokens)
        output, state = self.gru(vectors, initial_state=state)
        return output, state


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # For Eqn. (4), the  Bahdanau attention
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask, print_example_mask=False):
        w1_query = self.W1(query)
        w2_key = self.W2(value)
        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        if print_example_mask:
            print(
                "Query masks for the first setnence: ",
                tf.cast(query_mask[:1], tf.float32),
                "\nValue masks for the first setnence: ",
                tf.cast(value_mask[:1], tf.float32),
            )

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )

        return context_vector, attention_weights


class DecoderInput(typing.NamedTuple):
    new_tokens: Any
    enc_output: Any
    mask: Any


class DecoderOutput(typing.NamedTuple):
    logits: Any
    attention_weights: Any


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(
            self.output_vocab_size, embedding_dim
        )
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.attention = BahdanauAttention(self.dec_units)
        self.Wc = tf.keras.layers.Dense(
            dec_units, activation=tf.math.tanh, use_bias=False
        )
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)

    def call(self, inputs: DecoderInput, state=None) -> Tuple[DecoderOutput, tf.Tensor]:

        vectors = self.embedding(inputs.new_tokens)
        rnn_output, state = self.gru(vectors, initial_state=state)

        context_vector, attention_weights = self.attention(
            query=rnn_output, value=inputs.enc_output, mask=inputs.mask
        )
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)
        attention_vector = self.Wc(context_and_rnn_output)
        logits = self.fc(attention_vector)

        return DecoderOutput(logits, attention_weights), state


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.name = "masked_loss"
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

    def __call__(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask
        return tf.reduce_sum(loss)


class TrainTranslator(tf.keras.Model):
    def __init__(
        self,
        embedding_dim,
        units,
        input_text_processor,
        output_text_processor,
        use_tf_function=True,
    ):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(input_text_processor.vocabulary_size(), embedding_dim, units)
        decoder = Decoder(output_text_processor.vocabulary_size(), embedding_dim, units)

        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.use_tf_function = use_tf_function
        self.shape_checker = ShapeChecker()

    def train_step(self, inputs):
        self.shape_checker = ShapeChecker()
        if self.use_tf_function:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)

    def _preprocess(self, input_text, target_text):

        # Convert the text to token IDs
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)

        # Convert IDs to masks.
        input_mask = input_tokens != 0

        target_mask = target_tokens != 0

        return input_tokens, input_mask, target_tokens, target_mask

    def _train_step(self, inputs):
        input_text, target_text = inputs

        (input_tokens, input_mask, target_tokens, target_mask) = self._preprocess(
            input_text, target_text
        )

        max_target_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:
            # Encode the input
            enc_output, enc_state = self.encoder(input_tokens)

            # Initialize the decoder's state to the encoder's final state.
            # This only works if the encoder and decoder have the same number of
            # units.
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(max_target_length - 1):
                # Pass in two tokens from the target sequence:
                # 1. The current input to the decoder.
                # 2. The target for the decoder's next prediction.
                new_tokens = target_tokens[:, t: t + 2]
                step_loss, dec_state = self._loop_step(
                    new_tokens, input_mask, enc_output, dec_state
                )
                loss = loss + step_loss

            # Average the loss over all non padding tokens.
            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

        # Apply an optimization step
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # Return a dict mapping metric names to current value
        return {"batch_loss": average_loss}

    @tf.function(
        input_signature=[
            [
                tf.TensorSpec(dtype=tf.string, shape=[None]),
                tf.TensorSpec(dtype=tf.string, shape=[None]),
            ]
        ]
    )
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        # Run the decoder one step.
        decoder_input = DecoderInput(
            new_tokens=input_token, enc_output=enc_output, mask=input_mask
        )

        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
        self.shape_checker(dec_result.logits, ("batch", "t1", "logits"))
        self.shape_checker(dec_result.attention_weights, ("batch", "t1", "s"))
        self.shape_checker(dec_state, ("batch", "dec_units"))

        # `self.loss` returns the total for non-padded tokens
        y = target_token
        y_pred = dec_result.logits
        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state


class BatchLogs(tf.keras.callbacks.Callback):
    def __init__(self, key):
        self.key = key
        self.logs = []

    def on_train_batch_end(self, n, logs):
        self.logs.append(logs[self.key])


class Translator(tf.Module):
    def __init__(self, encoder, decoder, input_text_processor, output_text_processor):
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

        self.output_token_string_from_index = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(),
            mask_token="",
            invert=True,
        )

        # The output should never generate padding, unknown, or start.
        index_from_string = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token=""
        )
        token_mask_ids = index_from_string(["", "[UNK]", "[START]"]).numpy()

        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool_)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = token_mask

        self.start_token = index_from_string(tf.constant("[START]"))
        self.end_token = index_from_string(tf.constant("[END]"))

    def tokens_to_text(self, result_tokens):
        result_text_tokens = self.output_token_string_from_index(result_tokens)
        result_text = tf.strings.reduce_join(result_text_tokens, axis=1, separator=" ")
        result_text = tf.strings.strip(result_text)
        return result_text

    def sample(self, logits, temperature):
        # token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]
        logits = tf.where(self.token_mask, -np.inf, logits)

        if temperature == 0.0:
            new_tokens = tf.argmax(logits, axis=-1)
        else:
            logits = tf.squeeze(logits, axis=1)
            new_tokens = tf.random.categorical(logits / temperature, num_samples=1)
        return new_tokens

    def translate_unrolled(
        self, input_text, *, max_length=50, return_attention=True, temperature=1.0
    ):
        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text)
        enc_output, enc_state = self.encoder(input_tokens)

        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.start_token)

        result_tokens = []
        attention = []
        done = tf.zeros([batch_size, 1], dtype=tf.bool)

        for _ in range(max_length):
            dec_input = DecoderInput(
                new_tokens=new_tokens, enc_output=enc_output, mask=(input_tokens != 0)
            )

            dec_result, dec_state = self.decoder(dec_input, state=dec_state)

            attention.append(dec_result.attention_weights)

            new_tokens = self.sample(dec_result.logits, temperature)

            # If a sequence produces an `end_token`, set it `done`
            done = done | (new_tokens == self.end_token)
            # Once a sequence is done it only produces 0-padding.
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

            # Collect the generated tokens
            result_tokens.append(new_tokens)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        # Convert the list of generates token ids to a list of strings.
        result_tokens = tf.concat(result_tokens, axis=-1)
        result_text = self.tokens_to_text(result_tokens)

        if return_attention:
            attention_stack = tf.concat(attention, axis=1)
            return {"text": result_text, "attention": attention_stack}
        else:
            return {"text": result_text}


def load_model(
    input_processor: TextVectorization,
    output_processor: TextVectorization,
    weights: str,
    embedding_dim: int,
    n_units: int,
) -> Translator:

    # Instantiate seq-to-seq model
    seq_seq_model = TrainTranslator(
        embedding_dim=embedding_dim,
        units=n_units,
        input_text_processor=input_processor,
        output_text_processor=output_processor,
    )
    # Load trained weights

    seq_seq_model.load_weights(weights).expect_partial()

    # Instantiate translation object
    translator = Translator(
        encoder=seq_seq_model.encoder,
        decoder=seq_seq_model.decoder,
        input_text_processor=input_processor,
        output_text_processor=output_processor,
    )
    return translator

import numpy as np
import tensorflow as tf

from translator_model import __version__ as _version
from translator_model.config.base import config
from translator_model.utilities import model_tools as mt
from translator_model.utilities.data_manager import load_processor


def test_enlish_normalizer(test_input_data):
    # Given

    assert test_input_data[0][0] == "Is this the first record of the Set?"
    test_object = mt.english_normalizer(test_input_data[0][0]).numpy().decode()

    # Then
    assert test_object == "[START] is this the first record of the set [END]"


def test_arabic_normalizer(test_input_data):
    # Given

    assert test_input_data[1][0] == "هل هذهِ أول وثيقة المجموعة؟"
    test_object = mt.arabic_normalizer(test_input_data[1][0]).numpy().decode()

    # Then
    assert test_object == "[START] هل هذه اول وثيقة المجموعة [END]"


input_text_processor = inp_processor = load_processor(
    file_name=f"{config.app_config.inp_processor_save_file}{_version}.pkl",
    normalizer="english",
)
output_text_processor = targ_processor = load_processor(
    file_name=f"{config.app_config.targ_processor_save_file}{_version}.pkl",
    normalizer="arabic",
)


def test_input_preprocessor(test_input_data):
    # Given
    dataset = tf.data.Dataset.from_tensor_slices(
        (test_input_data[0], test_input_data[1])
    ).shuffle(len(test_input_data[0]), seed=config.model_config.tf_seed)
    for example, _ in dataset:
        break

    vocab = input_text_processor.get_vocabulary()[:10]
    tokens = input_text_processor(example)
    reconstructed = " ".join(
        list(np.array(input_text_processor.get_vocabulary())[tokens.numpy().tolist()])
    )

    # Then
    assert vocab == [
        "",
        "[UNK]",
        "[START]",
        "[END]",
        "i",
        "you",
        "the",
        "to",
        "a",
        "is",
    ]
    assert (
        tokens.dtype == tf.int64
        and tokens.shape == 10
        and tokens.numpy().tolist() == [2, 41, 13, 5, 33, 964, 1, 112, 116, 3]
    )
    assert reconstructed == "[START] why do you like wearing [UNK] so much [END]"


def test_output_preprocessor(test_input_data):
    # Given
    dataset = tf.data.Dataset.from_tensor_slices(
        (test_input_data[0], test_input_data[1])
    ).shuffle(len(test_input_data[0]), seed=config.model_config.tf_seed)
    for _, example in dataset:
        break

    vocab = output_text_processor.get_vocabulary()[:10]
    tokens = output_text_processor(example)
    print(tokens)
    reconstructed = " ".join(
        list(np.array(output_text_processor.get_vocabulary())[tokens.numpy().tolist()])
    )

    # Then
    assert vocab == [
        "",
        "[UNK]",
        "[START]",
        "[END]",
        "ان",
        "توم",
        "من",
        "لا",
        "في",
        "هل",
    ]
    assert (
        tokens.dtype == tf.int64
        and tokens.shape == 7
        and tokens.numpy().tolist() == [2, 62, 90, 1, 1, 71, 3]
    )
    assert reconstructed == "[START] لماذا تحب [UNK] [UNK] كثيرا [END]"


def apply_encoder(input):
    longest_sent = max(
        [len(input[i].numpy().decode().split(" ")) for i in range(len(input))]
    )

    tokens = input_text_processor(input)
    encoder = mt.Encoder(
        input_text_processor.vocabulary_size(),
        config.model_config.embedding_dim,
        config.model_config.units,
    )
    example_enc_output, example_enc_state = encoder(tokens)

    return tokens, example_enc_output, example_enc_state, longest_sent


def test_encoder_shape(test_input_data):
    # Given

    dataset = tf.data.Dataset.from_tensor_slices(
        (test_input_data[0], test_input_data[1])
    ).shuffle(len(test_input_data[0]), seed=config.model_config.tf_seed)
    for inp, _ in dataset.batch(
        batch_size=config.model_config.batch_size,
        num_parallel_calls=True,
        deterministic=True,
    ).take(1):
        break

    tokens, example_enc_output, example_enc_state, _ = apply_encoder(input=inp)

    # Then

    assert inp.shape == config.model_config.batch_size
    assert tokens.shape == [config.model_config.batch_size, tokens.shape[1]]
    assert example_enc_output.shape == [
        config.model_config.batch_size,
        tokens.shape[1],
        config.model_config.units,
    ]
    assert example_enc_state.shape == [
        config.model_config.batch_size,
        config.model_config.units,
    ]


def test_decoder_shape(test_input_data):

    # Given
    dataset = tf.data.Dataset.from_tensor_slices(
        (test_input_data[0], test_input_data[1])
    ).shuffle(len(test_input_data[0]), seed=config.model_config.tf_seed)
    for inp, targ in dataset.batch(
        batch_size=config.model_config.batch_size,
        num_parallel_calls=True,
        deterministic=True,
    ).take(1):
        break

    enc_tokens, example_enc_output, example_enc_state, _ = apply_encoder(input=inp)

    tokens = output_text_processor(targ)
    start_index = output_text_processor.get_vocabulary().index("[START]")
    first_token = tf.constant([[start_index]] * tokens.shape[0])

    decoder = mt.Decoder(
        output_text_processor.vocabulary_size(),
        config.model_config.embedding_dim,
        config.model_config.units,
    )

    dec_result, dec_state = decoder(
        inputs=mt.DecoderInput(
            new_tokens=first_token,
            enc_output=example_enc_output,
            mask=(enc_tokens != 0),
        ),
        state=example_enc_state,
    )
    sampled_tokens = tf.random.categorical(
        dec_result.logits[:, 0, :], num_samples=1, seed=config.model_config.tf_seed
    )

    # Then

    assert dec_result.logits.shape == [
        config.model_config.batch_size,
        1,
        output_text_processor.vocabulary_size(),
    ]

    assert dec_state.shape == [
        config.model_config.batch_size,
        config.model_config.units,
    ]

    assert all(np.array(output_text_processor.get_vocabulary())[sampled_tokens.numpy()])


def test_attention_layer(test_input_data):

    # Given
    dataset = tf.data.Dataset.from_tensor_slices(
        (test_input_data[0], test_input_data[1])
    ).shuffle(len(test_input_data[0]), seed=config.model_config.tf_seed)

    for inp, _ in dataset.batch(
        batch_size=config.model_config.batch_size,
        num_parallel_calls=True,
        deterministic=True,
    ).take(1):

        tokens, example_enc_output, example_enc_state, _ = apply_encoder(input=inp)

    attention_layer = mt.BahdanauAttention(config.model_config.units)
    attention_dim_2 = 2
    attention_dim_3 = 10
    example_attention_query = tf.random.normal(
        shape=[config.model_config.batch_size, attention_dim_2, attention_dim_3]
    )

    context_vector, attention_weights = attention_layer(
        query=example_attention_query,
        value=example_enc_output,
        mask=(tokens != 0),
        print_example_mask=True,
    )

    # Then

    assert context_vector.shape == [
        config.model_config.batch_size,
        attention_dim_2,
        config.model_config.units,
    ]
    assert attention_weights.shape == [
        config.model_config.batch_size,
        attention_dim_2,
        tokens.shape[1],
    ]
    assert np.isclose(a=np.sum(attention_weights[:1, 0, :]), b=1, atol=0.001)

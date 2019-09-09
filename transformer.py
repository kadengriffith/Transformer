'''
This is a sequence to sequence approach to using the Transformer model.
'''

from __future__ import absolute_import, unicode_literals

# TensorFlow
import tensorflow as tf
# TensorFlow Datasets for encoding
import tensorflow_datasets as tfds

# The Transformer model
from model.Transformer import Transformer
from model.CustomSchedule import CustomSchedule
from model.network import create_masks, loss_function

# Utilities
import random
import json
import time
import os
import numpy as np

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

CONFIG_FILE = os.path.join(DIR_PATH, "config.json")

with open(CONFIG_FILE, encoding='utf-8-sig') as fh:
    data = json.load(fh)

settings = dict(data)

CKPT_MODEL = settings["checkpoint"]

ENCODE_METHOD = tfds.features.text.SubwordTextEncoder

# Data constraints
MAX_LENGTH = 40

# Hyperparameters
NUM_LAYERS = settings["layers"]
D_MODEL = settings["model-depth"]
DFF = settings["dff"]
NUM_HEADS = settings["attn-heads"]
DROPOUT = settings["dropout"]

# Training settings
EPOCHS = settings["epochs"]
BATCH_SIZE = settings["batch-size"]

# Adam optimizer params
BETA_1 = 0.95
BETA_2 = 0.99
EPSILON = 1e-9

CONTINUE_FROM_CKPT = False
MODEL_NAME = f"t_{NUM_LAYERS}_{NUM_HEADS}_{D_MODEL}_{DFF}_{int(time.time())}"

if not CKPT_MODEL == False:
    # If a model name is given train from that model
    CONTINUE_FROM_CKPT = True

    MODEL_NAME = CKPT_MODEL

    CHECKPOINT_PATH = os.path.join(DIR_PATH,
                                   f"checkpoints/{CKPT_MODEL}/")

MODEL_PATH = os.path.join(DIR_PATH,
                          f"checkpoints/{MODEL_NAME}/")

# Random seed for repeatability
SEED = settings["seed"]
random.seed(SEED)
tf.random.set_seed(SEED)


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


def print_epoch(what, clear=False):
    # Overwrite the line to see live updated results
    print(f"{what}\r", end="")

    if clear:
        # Clear the line being overwritten by print_epoch
        print("\n")


if __name__ == "__main__":
    print("Starting Transformer training...")

    train_X = []
    train_y = []

    # AND Boolean logic
    examples = [([0, 0], 0),
                ([0, 1], 0),
                ([1, 0], 0),
                ([1, 1], 1)]

    print(f"Shuffling data with seed: {SEED}\n")
    random.shuffle(examples)

    # Get training examples
    for example in examples:
        try:
            Xs, ys = example

            train_X.append(str(Xs))
            train_y.append(str(ys))
        except:
            pass

    assert len(train_X) == len(train_y)

    print(f"Set to train with {len(train_X)} examples.\n")

    print("Building vocabulary...\n")

    # Convert arrays to TensorFlow constants
    train_X_const = tf.constant(train_X)
    train_y_const = tf.constant(train_y)

    # Turn the constants into TensorFlow Datasets
    training_dataset = tf.data.Dataset.from_tensor_slices((train_X_const,
                                                           train_y_const))

    tokenizer_X = ENCODE_METHOD.build_from_corpus((X.numpy() for X, _ in training_dataset),
                                                  target_vocab_size=2**13)

    tokenizer_y = ENCODE_METHOD.build_from_corpus((y.numpy() for _, y in training_dataset),
                                                  target_vocab_size=2**13)

    print("\nEncoding inputs...")

    def encode(lang1, lang2):
        lang1 = [tokenizer_X.vocab_size] + tokenizer_X.encode(
            lang1.numpy()) + [tokenizer_X.vocab_size + 1]

        lang2 = [tokenizer_y.vocab_size] + tokenizer_y.encode(
            lang2.numpy()) + [tokenizer_y.vocab_size + 1]

        return lang1, lang2

    def tf_encode(txt, eq):
        return tf.py_function(encode, [txt, eq], [tf.int64, tf.int64])

    training_dataset = training_dataset.map(tf_encode)

    training_dataset = training_dataset.filter(filter_max_length)

    # Cache the dataset to memory to get a speedup while reading from it.
    training_dataset = training_dataset.cache()

    # Batch the data
    training_dataset = training_dataset.padded_batch(BATCH_SIZE,
                                                     padded_shapes=([-1], [-1]))

    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    input_vocab_size = tokenizer_X.vocab_size + 2
    target_vocab_size = tokenizer_y.vocab_size + 2

    print("...done.")
    print("\nDefining the Transformer model...")

    # Using the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(D_MODEL),
                                         beta_1=BETA_1,
                                         beta_2=BETA_2,
                                         epsilon=EPSILON)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(
        name="train_acc")

    transformer = Transformer(NUM_LAYERS,
                              D_MODEL,
                              NUM_HEADS,
                              DFF,
                              input_vocab_size,
                              target_vocab_size,
                              DROPOUT)

    print("...done.")
    print("\nTraining...\n")

    # Model saving
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    if CONTINUE_FROM_CKPT:
        # Load last checkpoint
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  CHECKPOINT_PATH,
                                                  max_to_keep=999)
    else:
        if not os.path.isdir(f"checkpoints"):
            os.mkdir(f"checkpoints")

        if not os.path.isdir(f"checkpoints/{MODEL_NAME}"):
            os.mkdir(f"checkpoints/{MODEL_NAME}")

        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  MODEL_PATH,
                                                  max_to_keep=999)

    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint and CONTINUE_FROM_CKPT:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f"Restored from {CHECKPOINT_PATH} checkpoint!\n")

    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
                                                                         tar_inp)

        with tf.GradientTape() as tape:
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)

            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss,
                                  transformer.trainable_variables)

        optimizer.apply_gradients(zip(gradients,
                                      transformer.trainable_variables))

        train_loss(loss)
        train_acc(tar_real, predictions)

    # Train
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_acc.reset_states()

        for (batch, (inp, tar)) in enumerate(training_dataset):
            train_step(inp, tar)

            if batch % 10 == 0:
                print_epoch("Epoch {}/{} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1,
                    EPOCHS,
                    batch,
                    train_loss.result(),
                    train_acc.result()))

        print_epoch("Epoch {}/{} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
            epoch + 1,
            EPOCHS,
            batch,
            train_loss.result(),
            train_acc.result()), clear=True)

        # Calculate the time the epoch took to complete
        # The first epoch seems to take significantly longer than the others
        print(f"Epoch took {int(time.time() - start)}s\n")

        if epoch == (EPOCHS - 1):
            # Save a checkpoint of model weights
            ckpt_save_path = ckpt_manager.save()
            print(f'Saved {MODEL_NAME} to {ckpt_save_path}\n')

            # Delete old config
            os.remove(CONFIG_FILE)

            settings["checkpoint"] = MODEL_NAME

            # Write the config to use the checkpoint on next run
            with open(CONFIG_FILE, mode="w") as fh:
                json.dump(settings, fh)

            break

    print("...done.")

    def evaluate(inp_sentence):
        start_token = [tokenizer_X.vocab_size]
        end_token = [tokenizer_X.vocab_size + 1]

        inp_sentence = start_token + \
            tokenizer_X.encode(inp_sentence) + end_token

        encoder_input = tf.expand_dims(inp_sentence, 0)

        decoder_input = [tokenizer_y.vocab_size]

        output = tf.expand_dims(decoder_input, 0)

        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input,
                                                                             output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # Select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # Return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, tokenizer_y.vocab_size + 1):
                return tf.squeeze(output, axis=0), attention_weights

            # Concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(sentence):
        result, attention_weights = evaluate(sentence)

        prediction = tokenizer_y.decode([i for i in result
                                         if i < tokenizer_y.vocab_size])

        return prediction

    print(f"\nTesting AND logic...\n")

    print(f"[0, 0] -> {translate('[0, 0]')}")
    print(f"[0, 1] -> {translate('[0, 1]')}")
    print(f"[1, 0] -> {translate('[1, 0]')}")
    print(f"[1, 1] -> {translate('[1, 1]')}")

    print(f"\n...done.")

    print("\nExiting.")

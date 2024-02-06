# Training Loop and Inference

"""
This file contains the training pipeline for a Transformer model specialized in 
melody generation. It includes functions to calculate loss, perform training steps,
and orchestrate the training process over multiple epochs. The script also demonstrates
the use of the MelodyGenerator class to generate a melody after training.

The training process uses a custom implementation of the Transformer model, defined in
the transformer.py module, and prepares data using the MelodyPreprocessor class from
melodypreprocessor.py

Global parameters such as the number of epochs, batch size and path to the dataset are
defined. The script supports dynamic padding of sequences and employs the Sparse
Categorical Crossentropy loss function for model training.

For simplicity's sake, training does not deal with masking of padded values in the encoder
and decoder. Also, look-ahead masking is not implemented.

Central Components:
- _calculate_loss_function: Computes the loss between the actual and predicted sequences
- _train_step: Executes a single training step, including forward pass and backpropogation
- train: runs the training loop over the entire dataset for a given number of epochs
- _right_pad_sequence_once: Utility function for padding sequences

The script concludes by instantiating the Transformer model, conducting the training,
and generating a sample melody using the trained model
"""

import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from melodyPreprocessor import MelodyPreprocessor
from transformer import Transformer
from melodyGenerator import MelodyGenerator
from keras.preprocessing.text import Tokenizer

# Global parameters
EPOCHS = 3
BATCH_SIZE = 32
DATA_PATH = "essenDataset.json"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 1500

# Loss function and optimizer
sparse_categorical_crossentropy = SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
optimizer = Adam()


def train(train_dataset, transformer, epochs):
    """
    Trains the transformer model on a given dataset for a specified number of epochs

    Parameters:
        train_dataset (tf.data.Dataset): The training dataset
        transformer (Transformer): The Transformer model instance
        epochs (int): The number of epochs to train the model
    """
    print("Training the model...")
    for epoch in range(epochs):
        total_loss = 0
        # Iterate over each batch in the training dataset
        for batch, (input, target) in enumerate(train_dataset):
            # Create masks
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                input, target
            )
            # Perform a single training step
            batch_loss = _train_step(
                input,
                target,
                transformer,
                enc_padding_mask,
                combined_mask,
                dec_padding_mask,
            )
            total_loss += batch_loss
            print(f"Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy()}")


@tf.function
def _train_step(
    input, target, transformer, enc_padding_mask, look_ahead_mask, dec_padding_mask
):
    """
    Performs a single training step for the Transformer model

    Parameters:
        input (tf.Tensor): The input sequences
        target (tf.Tensor): The target sequences
        transformer (Transformer): The Transformer model instance

    Returns:
        tf.Tensor: The loss value for the training step
    """
    # Prepare the target input and real output for the decoder
    # Pad the sequences on the right by one position

    target_input = _right_pad_sequence_once(target[:, :-1])
    target_real = _right_pad_sequence_once(target[:, 1:])

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        # Forward pass through the transformer model
        predictions = transformer(
            input,
            target_input,
            True,
            enc_padding_mask,
            look_ahead_mask,
            dec_padding_mask,
        )

        # Compute loss between the real output and the predictions
        loss = _calculate_loss(target_real, predictions)

    # Calculate gradients with respect to the model's trainable variables
    gradients = tape.gradient(loss, transformer.trainable_variables)

    # Apply gradients to update the model's paramters
    gradient_variable_pairs = zip(gradients, transformer.trainable_variables)
    optimizer.apply_gradients(gradient_variable_pairs)

    # Return the computed loss for this training step
    return loss


def _calculate_loss(real, pred):
    """
    Computes the loss between the real and predicted sequences

    Parameters:
        real (tf.Tensor): The actual target sequences
        pred (tf.Tensor): The predicted sequences by the model

    Returns:
        average_loss (tf.Tensor): The computed loss value
    """

    # Compute loss using the Sparse Categorical Crossentropy
    loss_ = sparse_categorical_crossentropy(real, pred)

    # Create a mask to filter out zeros (padded values) in the real sequences
    boolean_mask = tf.math.equal(real, 0)
    mask = tf.math.logical_not(boolean_mask)

    # Convert mask to the same dtype as the loss for multiplication
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Apply the mask to the loss, ignoring losses on padded positions
    loss_ *= mask

    # Calculate average loss, excluding the padded positions
    total_loss = tf.reduce_sum(loss_)
    number_of_non_padded_elements = tf.reduce_sum(mask)
    average_loss = total_loss / number_of_non_padded_elements

    return average_loss


def _right_pad_sequence_once(sequence):
    """
    Pads a sequence with a single zero at the end

    Parameters:
        sequence (tf.Tensor): The sequence to be padded

    Returns:
        tf.Tensor: The padded sequence
    """
    return tf.pad(sequence, [[0, 0], [0, 1]], "CONSTANT")


def create_masks(input, target):
    enc_padding_mask = create_padding_mask(input)
    dec_padding_mask = create_padding_mask(input)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


if __name__ == "__main__":
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    train_dataset = melody_preprocessor.create_training_dataset()
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        dropout_rate=0.1,
    )

    train(train_dataset, transformer_model, EPOCHS)

    transformer_model.save("essen_model_3_epochs_lookahead_mask")

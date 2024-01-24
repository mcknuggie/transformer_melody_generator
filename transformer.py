"""
This python script contains a Transformer model. The architecture has been adapted 
in this context for musical melody generation. The core mechanism of the transformer
is the concept of "attention".

This code makes use of an Encoder-Decoder architecture (and their associated layers).
It also uses a positional encoder so that the model is aware of the order/relative position
of different notes.

Central Components:
- Transformer: Central model class that combines the Encoder and Decoder classes.
- Encoder: Processes the input sequence and generates a context-rich representation.
- Decoder: Generates the output sequence based on the Encoder's output and its own input.
- EncoderLayer & DecoderLayer: Individual layers that make up the Encoder and Decoder
- get_angles and sinusoidal_positon_encoding: Functions the generate positional encoding
  based on the sequence length and model dimensionality

Usage:
To use the transformer model, instantiate it with the required dimensions, number of layers,
voabulary size, and other parameters. The model can then be used for training or inference
tasks in music generation or other sequence-to-sequence transformations.

Note:
This implementation of the transformer model is designed for flexibility and can be
adapted for various sequence-to-sequnce tasks beyond music generation.
"""

import numpy as np
import tensorflow as tf
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
)


def sinusoidal_positon_encoding(num_positions, d_model):
    """
    Compute poisitonal encoding for a given position and dimension

    Parameters:
        num_positions (int): Number of positions
        d_model (int): Dimension of the model

    Returns:
        Tensor: Positional encoding for the given position and dimension
    """

    angles = _get_angles(
        np.arange(num_positions)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )

    # Apply sin to even indices in the array, 2i
    sines = np.sin(angles[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angles[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]  # (1, position, d_model)

    return tf.cast(pos_encoding, dtype=tf.float32)


def _get_angles(pos, i, d_model):
    """
    Computer the angles for the positional encoding

    Parameters:
        pos (np.ndarray): Positions
        i (np.ndarray): Indices
        d_model (int): Dimension of the model

    Returns:
        np.ndarray: Angles for the positional encoding
    """
    angle_dropout_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_dropout_rates


class Transformer(tf.keras.Model):
    """
    The transformer model architecture, consisting of an Encoder and Decoder.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        input_vocab_size,
        target_vocab_size,
        max_num_positions_in_pe_encoder,
        max_num_positions_in_pe_decoder,
        dropout_rate=0.1,
    ):
        """
        Parameters:
            num_layers (int): Number of layers in both Encoder and Decoder
            d_model (int): Dimension of the model (Dimension of the embedding - number of
                columns in I, Q, K, and V)
            num_heads (int): Number of attention heads (Z = concat(Z1, Z2, Z3, ... , ZN))
            d_feedforward (int): Dimension of the feed forward network
                (dimension or size of the input layer of the neural network;
                number of input variables)
            input_vocab_size (int): Size of the input vocabulary (size of input sequence)
            target_vocab_size (int): Size of the target vocabulary (size of target sequence)
            max_num_positions_in_pe_encoder (int): The maximum positions for input
                (maximum length of a sequence that we can expect)
            max_num_positions_in_pe_decoder (int): The maximum positions for target
                (maximum length of a sequence that we can expect)
            dropout_rate (float): Dropout rate (probability that a node gets skipped in order
                to reduce the liklihood of overfitting)
        """

        super(Transformer, self).__init__()
        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            d_feedforward,
            input_vocab_size,
            max_num_positions_in_pe_encoder,
            dropout_rate,
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            d_feedforward,
            target_vocab_size,
            max_num_positions_in_pe_decoder,
            dropout_rate,
        )

        self.final_layer = Dense(target_vocab_size)
        """
        single Nueral Net Layer using activation/squishification function, 
        variables, weights, and a bias value.
        """

        # self.num_layers = num_layers
        # self.d_model = d_model
        # self.num_heads = num_heads
        # self.d_feedforward = d_feedforward
        # self.input_vocab_size = input_vocab_size
        # self.target_vocab_size = target_vocab_size
        # self.max_num_positions_in_pe_encoder = max_num_positions_in_pe_encoder
        # self.max_num_positions_in_pe_decoder = max_num_positions_in_pe_decoder
        # self.dropout_rate = dropout_rate

    def call(
        self,
        input,
        target,
        training,
        enc_padding_mask,
        look_ahead_mask,
        dec_padding_mask
    ):
        """
        Process the input through the Transformer model. (forward pass)

        Parameters:
            input (Tensor): Input tensor to the Encoder
            target (Tensor): Target tensor for the Decoder
            training (bool): Whether the layer should behave in training mode
                as opposed to inference)
            enc_padding_mask (Tensor): Padding mask for the Encoder
            look_ahead_mask (Tensor): Look-ahead mask for the Decoder
                (used in the Masked Multi-Head Attention block)
            dec_padding_mask (Tensor): Padding mask for the Decoder

        Returns:
            Tensor: The final output of the Transformer
            dict: Attention weights from the Decoder layers
        """

        # input = inputs[0]
        # target = inputs[1]
        # enc_padding_mask = mask[0]
        # look_ahead_mask = mask[1]
        # dec_padding_mask = mask[2]

        enc_output = self.encoder(
            input, training, enc_padding_mask
        )  # (batch_size, input_seq_len, d_model)

        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, dec_padding_mask
        )  # (batch_size, target_seq_len, d_model)

        logits = self.final_layer(
            dec_output
        )  # (batch_size, target_seq_len, target_vocab_size)
        """
        A logit is just a score or a probability that is associated with
        an item (word or a note) in a sequence. It is the probability that the 
        next item in the sequence will crysalize into the associated word/note.
        """

        return logits

    # def get_config(self):
    #     config = super(Transformer, self).get_config()
    #     config.update(
    #         {
    #             "num_layers": self.num_layers,
    #             "d_model": self.d_model,
    #             "num_heads": self.num_heads,
    #             "d_feedforward": self.d_feedforward,
    #             "input_vocab_size": self.input_vocab_size,
    #             "target_vocab_size": self.target_vocab_size,
    #             "max_num_positions_in_pe_encoder": self.max_num_positions_in_pe_encoder,
    #             "max_num_positions_in_pe_decoder": self.max_num_positions_in_pe_decoder,
    #             "dropout_rate": self.dropout_rate,
    #         }
    #     )
    #     return config


class Encoder(tf.keras.layers.Layer):
    """
    The Encoder of a transformer model, consisting of multiple EncoderLayers
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        input_vocab_size,
        maximum_positions_in_pe,
        dropout_rate=0.1,
    ):
        """
        Parameters
            num_layers (int): Number of EncoderLayers
            d_model (int): Dimension of the model
            d_feedforward (int): Dimension of the feed forward network
            input_vocab_size (int): Size of the input vocabulary
            maximum_positions_in_pe (int): The Maximum sequence length that this model
                might ever be used with
            dropout_rate (float): Dropout rate
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = sinusoidal_positon_encoding(
            maximum_positions_in_pe, d_model
        )
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, d_feedforward, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Process the input through the Encoder

        Args:
            x (Tensor): Input Tensor
            training (bool): Whether the layer should behave in training mode
            mask (Tensor): Mask to be applied on attention weights

        Returns:
            Tensor: Output of the Encoder
        """
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Perform Positional Encoding before EncoderLayer
        sliced_pos_encoding = self._get_sliced_positional_encoding(x)
        x += sliced_pos_encoding

        x = self.dropout(x, training=training)

        # Perform EncoderLayer num_layers times
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        # output of Encoder
        return x  # (batch_size, input_seq_len, d_model)

    def _get_sliced_positional_encoding(self, x):
        """
        Get a slice of the full positional ecnoding

        Parameters:
            x (Tensor): Input tensor

        Returns:
            Tensor: A slice of the full positional encoding
        """
        number_of_tokens = x.shape[1]
        return self.pos_encoding[:, :number_of_tokens, :]


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder Layer of a transformer, consisting of MutliHeadAttention and
    Feed Forward Neural Network
    """

    def __init__(self, d_model, num_heads, d_feedforward, dropout_rate=0.1):
        """
        Parameters:
            d_model (int): Dimension of the model
            num_heads (int): Number of attention heads
            d_feedforward (int): Dimension of the feed forward network
            dropout_rate (float): Dropout dropout_rate
        """
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(d_feedforward, activation="relu"), Dense(d_model)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Process the input through the Encoder layer. Contains the actual logic of the
        Encoder as seen in Figure 1 of "Attention is Everything".

        Parameter:
            x (Tensor): Input tensor
            training (bool): Whether the layer should behave in training mode
            mask (Tensor): Mask to be applied on attention weights

        Returns:
            Tensor: Output of the Encoder layer
        """
        attn_output = self.mha(
            x, x, x, attention_mask=mask
        )  # multi-head attention block
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # apply skip connection into Add & Norm

        ffn_output = self.ffn(out1)  # feed forward neural network block
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # apply skip connection into Add & Norm

        return out2


class Decoder(tf.keras.layers.Layer):
    """
    The Decoder of a Transformer model, consisting of multiple DecoderLayers
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        target_vocab_size,
        maximum_positions_in_pe,
        dropout_rate=0.1,
    ):
        """
        Parameters:
            num_layers (int): Number of DecoderLayers
            d_model (int): Dimension of the model (# columns of I, Q, K, V)
            num_heads (int): Number of attention heads
            d_feedforward (int): Dimension of the feed forward neural network
            target_vocab_size (int): Size of target vocabulary
            maximum_positions_in_pe (int): The maximum seuqence length that this model
                might ever be used with
            dropout_rate (float): Dropout rate (probabiliy a node is skipped/ignored to
                reduce liklihood of overfitting)
        """
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = sinusoidal_positon_encoding(
            maximum_positions_in_pe, d_model
        )

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, d_feedforward, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Process the input through the Decoder

        Parameters:
            x (Tensor): Input tensor to the Decoder
            enc_output (Tensor): Output from the Encoder
            training (bool): Whether the layer should behave in training mode
            look_ahead_mask (Tensor): Mask for the first MultiHeadAttention layer
            passing_mask (Tensor): Mask for the second MultiHeadAttention layer

        Returns:
            Tensor: The output of the Decoder
        """

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        sliced_pos_encoding = self._get_sliced_positional_encoding(x)
        x += sliced_pos_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )

        return x

    def _get_sliced_positional_encoding(self, x):
        """
        Get a slice of the full positional encoding

        Parameters:
            x (Tensor): Input tensor

        Returns:
            Tensor: A slice of the full positional encoding
        """
        number_of_tokens = x.shape[1]
        return self.pos_encoding[:, :number_of_tokens, :]


class DecoderLayer(tf.keras.layers.Layer):
    """
    Decoder Layer of a transformer, consistuing of two MultiHeadAttention layers
    and a Feed Forward Neural Network
    """

    def __init__(self, d_model, num_heads, d_feedforward, dropout_rate=0.1):
        """
        Parameters:
            d_model (int): Dimension of the model (# columns of I, Q, K, V)
            num_heads (int): Number of attention heads (Z)
            d_feedforward (int): Dimension of the feed forward network
            dropout_rate (float): Dropout rate
        """
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)

        self.ffn = tf.keras.Sequential(
            [Dense(d_feedforward, activation="relu"), Dense(d_model)]
        )

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Process the input through the Decoder layer

        Parameters:
            x (Tensor): Input tensor to the Decoder layer
            enc_output (Tensor): Output from the Encoder
            training (bool): Whether the layer should behave in training mode
            look_ahead_mask (Tensor): Mask for the first MultiHeadAttention layer
            padding_mask (Tensor): Mask for the second MultiHeadAttention layer

        Returns:
            Tensor: The output of the Decoder layer
        """

        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)  # skip connection

        attn2 = self.mha2(
            out1, enc_output, enc_output, attention_mask=padding_mask
        )  # Q: out1, K: encoder output, V: encoder output
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # skip connection

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # skip connection

        return out3


# END OF TRANSFORMER

if __name__ == "__main__":
    # Define Transformer parameters
    num_layers = 2
    d_model = 64
    num_heads = 2
    d_feedforward = 128
    input_vocab_size = 100
    target_vocab_size = 100
    dropout_dropout_rate = 0.1
    pe_input = 10
    pe_target = 10

    # Instantiate the Transformer model
    transformer_model = Transformer(
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        dropout_dropout_rate,
    )

    # Dummy input shapes for encoder and decoder
    dummy_inp = tf.random.uniform(
        (1, 10), dtype=tf.int64, minval=0, maxval=input_vocab_size
    )
    dummy_tar = tf.random.uniform(
        (1, 10), dtype=tf.int64, minval=0, maxval=target_vocab_size
    )

    # Build the model using dummy input
    transformer_model(
        dummy_inp,
        dummy_tar,
        training=False,
        enc_padding_mask=None,
        look_ahead_mask=None,
        dec_padding_mask=None,
    )

    # Display the model summary
    transformer_model.summary()

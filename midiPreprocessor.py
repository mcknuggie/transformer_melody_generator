# Prepares a dataset that will be ready for training

import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import TextVectorization


class MidiPreprocessor:
    """
    A class for preprocessing MIDI meldoies for a Transformer model

    This class takes Midi Tuples, tokenizes and encodes them, and
    prepares TensorFlow datasets for training sequence-to-sequnce
    models
    """

    def __init__(self, dataset_path, batch_size=32):
        """
        Initializes the MidiPreprocessor
        """

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.tokenizer = Tokenizer(filters="", lower=False, split=",")
        # self.tokenizer = TextVectorization()
        self.max_melody_length = None
        self.number_of_tokens = None

    @property
    def number_of_tokens_with_padding(self):
        """
        Returns the number of tokens in the vocabulary including padding.
        (the padding is the +1 on the end)

        Returns:
            int: The number of tokens in the vocabulary including padding
        """
        return self.number_of_tokens + 1

    def create_training_dataset(self):
        """
        Preprocesses the melody dataset and creates
        sequence-to-sequence training data

        Returns:
            tf_training_dataset: A TensorFlow dataset containing
            input-target pairs suitable for training a sequence-
            to-sequnce model
        """
        dataset = self._load_dataset()
        parsed_songs = [self._parse_melody(song) for song in dataset]
        cleaned_songs = [self._clean_melody(song) for song in parsed_songs]
        # print(cleaned_songs)
        separated_midi_songs = self._separate_midi_notes(cleaned_songs)
        # print(separated_midi_songs)
        tokenized_songs = self._tokenize_and_encode_melodies(separated_midi_songs)
        self._set_max_melody_length(tokenized_songs)
        self._set_number_of_tokens()
        input_sequences, target_sequences = self._create_sequence_pairs(
            tokenized_songs
        )
        tf_training_dataset = self._convert_to_tf_dataset(
            input_sequences, target_sequences
        )
        return tf_training_dataset

    def _separate_midi_notes(self, songs):     
        separated_songs = [] # a list of songs where each song is a list of notes but the inidividual note features are separated
        for song in songs:
            separated_song = [] # a single song represented by a bunch of individual note features
            for note in song:
                note_components = note.split()
                separated_song.extend(note_components)

            separated_songs.append(separated_song)

        return separated_songs

    def _load_dataset(self):
        """
        Loads the melody dataset from a text file

        Returns:
            list: A list of lines from the text file
        """

        with open(self.dataset_path, "r") as f:
            lines = f.readlines()
            return [line.strip() for line in lines]

    def _parse_melody(self, melody_str):
        """
        Parses a single melody string into a list of notes

        Parameters:
            melody_str (str): A string representation of a melody

        Returns:
            list: A list of notes extracted from the melody string
        """
        return melody_str.split("; ")

    def _clean_melody(self, note_list):
        """
        Removes () and , from melodies for easier processing

        Parameters:
            melody_str (str): A List of strs each representing a note

        Returns:
            list: a cleaned list of strs, each representing a clean note
        """
        cleaned_note_list = []

        for note_str in note_list:
            my_str = note_str.replace("(", "")
            my_str = my_str.replace(")", "")
            my_str = my_str.replace(",", "")
            cleaned_note_list.append(my_str)

        return cleaned_note_list

    def _tokenize_and_encode_melodies(self, melodies):
        """
        Tokenize and encodes a list of melodies.

        Parameters:
            melodies (list): A list of melodies (in other words: a list of lists of strings where each string contains a MIDI note) to be tokenized and encoded

        Returns:
            tokenized_melodies: A list of tokenized and encoded melodies
            (sequence of integers as opposed to a sequence of symbols)
        """

        self.tokenizer.fit_on_texts(melodies)
        tokenized_melodies = self.tokenizer.texts_to_sequences(melodies)
        return tokenized_melodies

    def _set_max_melody_length(self, melodies):
        """
        Sets the maximum length based on the dataset

        Parameters:
            melodies (list): A list of tokenized melodies
        """
        self.max_melody_length = max([len(melody) for melody in melodies])

    def _set_number_of_tokens(self):
        """
        Sets the number of tokens based on the tokenizer
        """
        self.number_of_tokens = len(self.tokenizer.word_index)

    def _create_sequence_pairs(self, melodies):
        """
        Creates input target pairs from tokenized melodies

        Parameters:
            melodies (list): A list of tokenized melodies

        Returns:
            tuple: Two numpy arrays representing
            input sequences and target sequences
        """
        input_sequences, target_sequences = [], []
        for melody in melodies:
            for i in range(1, len(melody)):
                input_seq = melody[:i]
                target_seq = melody[1 : i + 1]
                padded_input_seq = self._pad_sequence(input_seq)
                padded_target_seq = self._pad_sequence(target_seq)
                input_sequences.append(padded_input_seq)
                target_sequences.append(padded_target_seq)
        return np.array(input_sequences), np.array(target_sequences)

    def _pad_sequence(self, sequence):
        """
        Pads a sequence to the maximum sequence length

        Parameters:
            sequence (list): The sequence to be padded

        Returns:
            list: The padded sequence
        """
        return sequence + [0] * (self.max_melody_length - len(sequence))

    def _convert_to_tf_dataset(self, input_sequences, target_sequences):
        """
        Converts input and target sequences to a TensorFlow Dataset

        Parameters:
            input_sequences (list): Input sequences for the model
            target_sequences (list): Target sequences for the model

        Returns:
            batched_dataset (tf.data.Dataset): A batched and shuffled TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_sequences, target_sequences)
        )
        shuffled_dataset = dataset.shuffle(buffer_size=1000)
        batched_dataset = shuffled_dataset.batch(self.batch_size)
        return batched_dataset


if __name__ == "__main__":
    # Example
    preprocessor = MidiPreprocessor("prepped_midi/test_output.txt", batch_size=32)
    training_dataset = preprocessor.create_training_dataset()

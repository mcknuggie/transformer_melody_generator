from melodyGenerator import MelodyGenerator
from midiGenerator import MidiGenerator
from music21 import metadata, note, stream
from keras.preprocessing.text import Tokenizer
from keras.layers import TextVectorization
from keras import models
from transformer import Transformer
from melodyPreprocessor import MelodyPreprocessor
from midiPreprocessor import MidiPreprocessor

BATCH_SIZE = 32

def visualize_melody(melody):
    """
    Visualize a sequence of (pitch, duration) pairs using music21.

    Parameters:
        melody (str): A str of "pitch-duration" substrings separated by whitespaces.
    """

    note_list = melody.split()

    note_tuples = []

    for note_duration_str in note_list:
        temp_list = note_duration_str.split("-")
        pitch = temp_list[0]
        duration = int(float(temp_list[1]))
        note_tuples.append((pitch, duration))

    melody = note_tuples

    print("my melody: ", melody)

    score = stream.Score()
    score.metadata = metadata.Metadata(title="Transformer Melody")
    part = stream.Part()
    for n, d in melody:
        if n == "0":
            part.append(note.Rest(d))
        else:
            part.append(note.Note(n, quarterLength=d))
    score.append(part)
    score.show()

def generate_monophonic():
    data_path = "dataset.json"
    preprocessor = MelodyPreprocessor(data_path, batch_size=BATCH_SIZE)

    train_dataset = preprocessor.create_training_dataset()
    vocab_size = preprocessor.number_of_tokens_with_padding

    model = models.load_model(
        "essen_model_3_epochs_lookahead_mask",
        custom_objects={"Transformer": Transformer},
    )

    print("Generating a melody...")

    melody_generator = MelodyGenerator(model, preprocessor.tokenizer)
    start_sequence = ["B4-1.0", "0-1.0", "E4-1.0", "A4-2.0", "Bb4-1.0"]
    new_melody = melody_generator.generate(start_sequence)

    return new_melody

def generate_polyphonic():
    data_path = "prepped_midi/test_output.txt"
    preprocessor = MidiPreprocessor(data_path, batch_size=BATCH_SIZE)

    train_dataset = preprocessor.create_training_dataset()
    vocab_size = preprocessor.number_of_tokens_with_padding

    model = models.load_model(
        "test_midi_model",
        custom_objects={"Transformer": Transformer},
    )

    print("Generating a melody...")

    length_of_output = 50
    midi_generator = MidiGenerator(model, preprocessor.tokenizer, length_of_output)
    # start_sequence = ["(60, 100, 0, 450)", "(64, 100, 0, 450)", "(62, 100, 451, 900)", "(66, 100, 451, 900)"]
    # start_sequence = ["60 100 0 450", "64 100 0 450", "62 100 451 900", "66 100 451 900"]
    start_sequence = ["62", "56", "624", "1296", "65", "59", "600", "1272", "65", "48", "1368", "1680"]
    preprocessor.tokenizer.fit_on_texts(start_sequence)
    new_melody = midi_generator.generate(start_sequence)

    return new_melody



if __name__ == "__main__":

    new_melody = generate_monophonic()
    # new_melody = generate_polyphonic()

    print(f"Generated melody: {new_melody}")
    visualize_melody(new_melody)

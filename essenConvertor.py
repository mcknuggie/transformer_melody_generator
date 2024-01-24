from music21 import metadata, note, stream
import json


def load_dataset(dataset_path):
    """
    Loads the Essen dataset from a txt file

    Returns:
        list: A list of unformatted melodies and key signiatures from the dataset
    """

    storing_lines = False
    stored_lines = []
    current_line = ""

    with open(dataset_path, "r") as f:
        for line in f:
            if line.startswith("KEY"):
                storing_lines = True

            if ">>" in line:
                storing_lines = False
                current_line += line.strip()
                stored_lines.append(current_line.strip())
                current_line = ""

            elif storing_lines:
                current_line += line.strip()

    return stored_lines


def format_dataset(unformatted_data):
    """
    Formats a list of unformatted data

    Returns:
        list[tuples]: A list of tuples such that each tuple is as follows:

        (string: key_symbol, integer: octave_value, list[tuples]: (string: pitch (example: G4), integer: duration) )
    """

    formatted_data = []

    for song_str in unformatted_data:
        key_info, melody_info = song_str.split("MEL")

        key_info = key_info.split()
        octave_value = int(key_info[1])
        if octave_value > 5:
            octave_value = 4
        key_symbol = key_info[2]

        melody_info = melody_info.replace(" ", "")
        melody_info = melody_info[:-5]
        melody_info = melody_info[1:]
        melody = melody_string_to_notes(melody_info, key_symbol, octave_value)

        formatted_data.append((key_symbol, octave_value, melody))

    return formatted_data


def melody_string_to_notes(melody_str, key_symbol, octave_value):
    """
    Converts a single melody string into a list of playable notes

    Parameters:
        melody_str (string): A single string representing a musical melody
        key_symbol (string): A character representing the key of the melody
        ocatve_value (int): An integer that represents the root octave of the melody

    Returns:
        list[tuples]: each tuple is structured as follows -> (string: pitch (example: G4), integer: duration)
    """

    num_to_note = get_num_to_note(key_symbol)
    is_negative = False
    output_melody = []

    for char in melody_str:
        if char == "-":
            is_negative = True
        elif char.isnumeric():  # 0-9
            if char == "0":  # rest
                output_melody.append((None, 1))
            elif is_negative:
                is_negative = False
                adjusted_note_letter = str((int(char) * -1) + 8)
                output_melody.append(
                    (num_to_note[adjusted_note_letter] + str(octave_value - 1), 1)
                )
            else:
                output_melody.append((num_to_note[char] + str(octave_value), 1))
        elif char == "b" or char == "#":
            # add flat or sharp symbol on to pitch
            current_pitch = output_melody[-1][0]
            modified_pitch = current_pitch[:1] + char + current_pitch[1:]
            output_melody[-1] = (modified_pitch, output_melody[-1][1])
        elif char == "_":
            # add 1 to the duration of the previous note/rest
            output_melody[-1] = (output_melody[-1][0], output_melody[-1][1] + 1)
        elif char == ".":
            # multiply duration of previous note/rest by 1.5
            output_melody[-1] = (output_melody[-1][0], 1.5 * output_melody[-1][1])

    return output_melody


def notes_tuples_to_strings(formatted_data):
    """
    Convert note tuples into note strings
    """

    melody_strings = []

    for song in formatted_data:
        melody = song[2]
        melody_str = ""
        for note, duration in melody:
            if note is None:
                melody_str += "0" + "-" + str(float(duration)) + ", "
            else:
                melody_str += note + "-" + str(float(duration)) + ", "
        melody_strings.append(melody_str[:-2])

    return melody_strings


def write_to_json(output_array, file_path):
    """
    Writes an array of strings to a JSON file.

    Parameters:
        output_array (list): The array of strings to be written to the JSON file.
        file_path (str): The path to the JSON file.
    """
    data = {"outputs": output_array}

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)


def visualize_melody(melody):
    """
    Visualize a sequence of (pitch, duration) pairs using music21.

    Parameters:
        melody (list[tuples]): A list of tuples such that: (string: pitch, int: duration)
    """

    score = stream.Score()
    score.metadata = metadata.Metadata(title="Transformer Melody")
    part = stream.Part()
    for n, d in melody:
        if n is None:
            part.append(note.Rest(d))
        else:
            part.append(note.Note(n, quarterLength=d))
    score.append(part)
    score.show()


def get_num_to_note(key_symbol):
    match key_symbol:
        case "C":
            return {
                "1": "C",
                "2": "D",
                "3": "E",
                "4": "F",
                "5": "G",
                "6": "A",
                "7": "B",
            }
        case "D":
            return {
                "1": "D",
                "2": "E",
                "3": "F",
                "4": "G",
                "5": "A",
                "6": "B",
                "7": "C",
            }
        case "E":
            return {
                "1": "E",
                "2": "F",
                "3": "G",
                "4": "A",
                "5": "B",
                "6": "C",
                "7": "D",
            }
        case "F":
            return {
                "1": "F",
                "2": "G",
                "3": "A",
                "4": "B",
                "5": "C",
                "6": "D",
                "7": "E",
            }
        case "G":
            return {
                "1": "G",
                "2": "A",
                "3": "B",
                "4": "C",
                "5": "D",
                "6": "E",
                "7": "F",
            }
        case "A":
            return {
                "1": "A",
                "2": "B",
                "3": "C",
                "4": "D",
                "5": "E",
                "6": "F",
                "7": "G",
            }
        case "B":
            return {
                "1": "B",
                "2": "C",
                "3": "D",
                "4": "E",
                "5": "F",
                "6": "G",
                "7": "A",
            }
        case _:
            return None


if __name__ == "__main__":
    # Example
    unformatted_data = load_dataset("altdeu.txt")
    formatted_data = format_dataset(unformatted_data)
    melody_strings = notes_tuples_to_strings(formatted_data)

    melody_lens = []
    for melody_str in melody_strings:
        melody_lens.append(len(melody_str))

    print(max(melody_lens))

    # write_to_json(melody_strings, "essenDataset.json")

    # for song in melody_strings:
    #     print(song, "\n")

    # visualize_melody(formatted_data[0][2])

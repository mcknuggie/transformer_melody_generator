from mido import MidiFile
import os

# class Note:
#     def __init__(self, note, velocity, start_time, end_time):
#         """
#         Initializes a Note
#         """

#         self.note = note
#         self.velocity = velocity
#         self.start_time = start_time
#         self.end_time = end_time


def process_file(file_path):
    # Replace this function with your custom logic for each file
    print(get_note_information(file_path))


def process_folder(folder_path):
    # Iterate through all files in the current folder
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            process_file(file_path)


def get_note_information(midi_file_path):
    """
    Input: a MIDI file
    outputs: A list of lists. Each sublist contains notes where each note is as follows:
    Note(note=60, velocity=100, start_time=244, end_time=30)

    start_time = number of ticks into the file when the note is first pressed
    end_time = number of ticks into the file when the note is let go

    For now - we will move all music to the beginning of the file, such that the first note is always played on the first tick.
    """
    midi_file = MidiFile(midi_file_path)

    midi_tracks = []

    for i, track in enumerate(midi_file.tracks):
        print(track)
        combined_notes = []
        currently_playing_notes = {}
        elapsed_ticks = 0
        for msg in track:
            # msg.type  msg.note  msg.velocity  msg.time
            elapsed_ticks += msg.time
            if msg.type == "note_on":
                if msg.velocity != 0:
                    currently_playing_notes[msg.note] = (msg.velocity, msg.time)
                else:
                    velocity, start_time = currently_playing_notes[msg.note]
                    combined_notes.append((msg.note, velocity, start_time, elapsed_ticks))
                    currently_playing_notes[msg.note] = ()
            if msg.type == "note_off":
                velocity, start_time = currently_playing_notes[msg.note]
                combined_notes.append((msg.note, velocity, start_time, elapsed_ticks))
                currently_playing_notes[msg.note] = ()
        if combined_notes:
            midi_tracks.append(combined_notes)

    return midi_tracks


def write_tuples_to_file(tuples, file_path):
    with open(file_path, "w") as file:
        for t in tuples:
            line = ", ".join(
                map(str, t)
            )  # Convert each element to string and join them with a comma
            file.write(line + "\n")


def write_track_to_file(input_file_path, output_file_path):
    midi_file = MidiFile(input_file_path)
    with open(output_file_path, "w") as file:
        for i, track in enumerate(midi_file.tracks):
            for msg in track:
                line = str(msg)
                file.write(line + "\n")


if __name__ == "__main__":

    folder_to_process = "test_dataset_folder"

    # process_folder(folder_to_process)

    # note_info = get_note_information("test_dataset_folder/subfolder_2/test.mid")[0]

    print(get_note_information("twinkle.mid")[0])

    # write_track_to_file("test_dataset_folder/subfolder_2/test.mid", "output.txt")

    # write_tuples_to_file(note_info, "output2.txt")

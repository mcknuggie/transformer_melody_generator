from mido import MidiFile
import os


def get_note_information(midi_file_path):
    """
    Input: a MIDI file
    outputs: A list of lists. Each sublist contains notes where each note is as follows:
    Note(note=60, velocity=100, start_time=244, end_time=330)

    start_time = number of ticks into the file when the note is first pressed
    end_time = number of ticks into the file when the note is let go

    For now - we will move all music to the beginning of the file, such that the first note is always played on the first tick.
    """
    midi_file = MidiFile(midi_file_path)

    midi_tracks = []

    for i, track in enumerate(midi_file.tracks):
        combined_notes = []
        currently_playing_notes = {}
        elapsed_ticks = 0

        for msg in track:
            # msg.type  msg.note  msg.velocity  msg.time
            if msg.type in ["note_on", "note_off"]:
                elapsed_ticks += msg.time
            if msg.type == "note_on":
                if msg.velocity != 0:
                    if msg.note in currently_playing_notes: # if, in error, note is played twice in a row without letting go
                        velocity, start_time = currently_playing_notes[msg.note]
                        combined_notes.append((msg.note, velocity, start_time, elapsed_ticks))
                    currently_playing_notes[msg.note] = (msg.velocity, elapsed_ticks) # write/overwrite existing note
                else:  # if the velocity is 0, treat like note_off
                    if msg.note in currently_playing_notes:
                        velocity, start_time = currently_playing_notes[msg.note]
                        combined_notes.append((msg.note, velocity, start_time, elapsed_ticks))
                        del currently_playing_notes[msg.note]
            elif msg.type == "note_off":
                if msg.note in currently_playing_notes:
                    velocity, start_time = currently_playing_notes[msg.note]
                    combined_notes.append((msg.note, velocity, start_time, elapsed_ticks))
                    del currently_playing_notes[msg.note]
        if combined_notes:
            midi_tracks.append(combined_notes)

    return midi_tracks

def write_tuples_to_file(tuples, file_path):
    with open(file_path, "a") as file:
        for t in tuples:
            line = "; ".join(
                map(str, t)
            )  # Convert each element to string and join them with a semi-colon
            file.write(line + "\n")


# def write_midi_info_to_file(input_file_path, output_file_path):
#     midi_file = MidiFile(input_file_path)
#     with open(output_file_path, "w") as file:
#         for i, track in enumerate(midi_file.tracks):
#             for msg in track:
#                 line = str(msg)
#                 file.write(line + "\n")


def process_folder(folder_path):
    # Iterate through all files in the current folder
    for root, dirs, files in os.walk(folder_path):
        for i in range(len(files)):
            file_path = os.path.join(root, files[i])
            tuples = get_note_information(file_path)
            write_tuples_to_file(tuples, "prepped_midi/test_output.txt")

if __name__ == "__main__":

    # To run on entire dataset folder
    folder_to_process = "test_dataset_folder"
    process_folder(folder_to_process)

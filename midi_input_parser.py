# import midi

# # Open the default MIDI input port
# midi_in = midi.Input()

# while True:
#     # Get MIDI events from the input port
#     midi_events = midi_in.read(1024)

#     for event in midi_events:
#         # Check if event is a note on/off message
#         if event[0] == 144 or event[0] == 128:
#             note_number = event[1]
#             velocity = event[2]

#             # Parse note info
#             note = {"number": note_number, "velocity": velocity}

#             print("Note:", note)

import mido
from "python-rtmidi" import rtmidi

# Open the default MIDI input port
midi_in = mido.open_input()

for msg in midi_in:
    # Check if message is a note on/off message
    if msg.type == "note_on" or msg.type == "note_off":
        note_number = msg.note
        velocity = msg.velocity

        # Parse note info
        note = {"number": note_number, "velocity": velocity}

        print("Note:", note)


# import mido


# def print_midi_messages(device_name):
#     try:
#         with mido.open_input(device_name) as port:
#             print(f"Listening to MIDI controller: {device_name}")
#             for message in port:
#                 print(message)
#     except IOError:
#         print(f"Cannot open MIDI input: {device_name}")


# if __name__ == "__main__":
#     # Replace 'Your MIDI Device Name' with the name of your MIDI controller.
#     # You can find the name by running 'mido.get_input_names()' and selecting your MIDI device.
#     midi_device_name = "Your MIDI Device Name"
#     print_midi_messages(midi_device_name)



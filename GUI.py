import pygame
import os
import sys
import customtkinter as ctk
import threading

# Initialize Pygame mixer
pygame.mixer.init()

# Function to play an MP3 file
def play_mp3(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

# Function to handle the playing of songs in a separate thread
def handle_songs(folder_path, play_next_event):
    song_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
    if not song_files:
        print("No MP3 files found in the folder.")
        sys.exit()

    current_song = 0
    play_mp3(os.path.join(folder_path, song_files[current_song]))

    while True:
        play_next_event.wait()  # Wait for the signal to play the next song
        play_next_event.clear()

        # Move to next song
        current_song = (current_song + 1) % len(song_files)
        play_mp3(os.path.join(folder_path, song_files[current_song]))

# Set up CustomTkinter
ctk.set_appearance_mode("dark")
app = ctk.CTk()
app.title("AFK DJ")
app.geometry("450x150")

# Event for signaling when to play the next song
play_next_event = threading.Event()

# Button callback functions
def play_callback():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.pause()
    else:
        pygame.mixer.music.unpause()

def next_callback():
    play_next_event.set()

# GUI Buttons
play_button = ctk.CTkButton(app, text="Play/Pause", command=play_callback)
play_button.grid(row=0, column=0, padx=20, pady=20)

next_button = ctk.CTkButton(app, text="Next", command=next_callback)
next_button.grid(row=0, column=1, padx=20, pady=20)

# Start the song handler thread
song_thread = threading.Thread(target=handle_songs, args=('songs', play_next_event))
song_thread.daemon = True
song_thread.start()

# Start the Tkinter event loop
app.mainloop()

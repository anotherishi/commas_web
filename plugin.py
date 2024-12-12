import whisper
import parselmouth
from parselmouth.praat import call
import subprocess
import re
from transformers import pipeline
from TTS.api import TTS
import wave
from pydub import AudioSegment
import librosa
import numpy as np
from scipy.spatial.distance import cdist
import cv2
import mediapipe as mp
import numpy as np
from google.colab import files
import time

model = whisper.load_model("small")

#INPUT FILE IS AUDIO.M4A. audio.wav should NOT be changed.
# subprocess.run(["ffmpeg", "-i", "audio.m4a", "audio.wav"])

result = model.transcribe("audio.wav")
transcription = result['text']
#OUTPUTS TRANSCRIPTION
print(f"Transcription: {transcription}")

corrector = pipeline("text2text-generation", model="pszemraj/flan-t5-large-grammar-synthesis")

transcript = transcription

lines = transcript.strip().split('\n')

corrected_lines = []
for i, line in enumerate(lines):
    corrected_text = corrector(line)[0]['generated_text']
    corrected_lines.append(corrected_text)

corrected_transcript = "\n".join(corrected_lines)
#OUTPUTS THE CORRECTED TRANSCRIPT
print("\n🎉 **Corrected Transcript**:\n")
print(corrected_transcript)

def check_errors(correct_lines, incorrect_lines):
    for i, (correct, incorrect) in enumerate(zip(correct_lines, incorrect_lines)):
        if correct != incorrect:
            print(f"Error in line {i + 1}:")
            print(f"Incorrect: {incorrect}")
            print(f"Expected: {correct}")

#OUTPUTS THE LINES HAVING GRAMMATICAL ERRORS ALONG WITH THE CORRECT LINES
check_errors(lines, corrected_lines)

sound = parselmouth.Sound("audio.wav")
duration = call(sound, "Get total duration")
silences = call(sound, "To TextGrid (silences)", 100, 0.1, -25, 0.1, 0.05, "silent", "sounding")
n_pauses = call(silences, "Count intervals where", 1, "is equal to", "silent")
n_words = len(transcription.split())
speaking_rate = n_words / duration 

#OUTPUTS TOTAL DURATION, NUMBER OF PAUSES, SPEAKING RATE
print(f"Total Duration: {duration} seconds")
print(f"Number of Pauses: {n_pauses}")
print(f"Speaking Rate: {speaking_rate} words/second")

filler_words = ["uh", "um", "like", "you know", "er", "ah", "so", "actually", "basically"]

pattern = r'\b(' + '|'.join(filler_words) + r')\b'

matches = re.findall(pattern, transcription, flags=re.IGNORECASE)

print(f"Filler words found: {matches}")

from collections import Counter
word_counts = Counter(matches)
#OUTPUTS A PYTHON DICTIONARY WHOSE KEYS ARE WORDS AND VALUES ARE NUMBER OF TIMES THE WORDS ARE USED
print(f"Filler word counts: {word_counts}")

tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DCA")

def synthesize_audio_from_text(text_input, output_file):
    # Generate synthesized audio using FastSpeech2
    tts_model.tts_to_file(text=text_input, file_path=output_file)
    return output_file

def compare_audio_files(original_audio_file, synthesized_audio_file):
    original_audio, sr_original = librosa.load(original_audio_file, sr=None)
    synthesized_audio, sr_synthesized = librosa.load(synthesized_audio_file, sr=None)

    mfcc_original = librosa.feature.mfcc(y=original_audio, sr=sr_original, n_mfcc=13)
    mfcc_synthesized = librosa.feature.mfcc(y=synthesized_audio, sr=sr_synthesized, n_mfcc=13)

    distance = cdist(mfcc_original.T, mfcc_synthesized.T, 'cosine')

    print(f"Average Cosine Distance between Original and Synthesized MFCCs: {np.mean(distance)}")
    if np.mean(distance) > 0.5:  
        print("Pronunciation error detected!")
    else:
        print("Pronunciation is correct.")

    speech_rate_original = len(original_audio) / sr_original
    speech_rate_synthesized = len(synthesized_audio) / sr_synthesized
    print(f"Original Speech Rate: {speech_rate_original:.2f} sec")
    print(f"Synthesized Speech Rate: {speech_rate_synthesized:.2f} sec")

    pitch_original = librosa.yin(original_audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    pitch_synthesized = librosa.yin(synthesized_audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))

    pitch_variation_original = np.std(pitch_original)
    pitch_variation_synthesized = np.std(pitch_synthesized)

    print(f"Pitch Variation Original: {pitch_variation_original:.2f}")
    print(f"Pitch Variation Synthesized: {pitch_variation_synthesized:.2f}")


transcription_text = transcription
original_audio_input = "audio.wav" #PATH FOR ORIGINAL AUDIO IN .WAV FORMAT
synthesized_audio_output = "synthesized_audio.wav"  #PATH FOR SYNTHESIZED AUDIO
#IT CREATES A NEW FILE
synthesized_audio_file = synthesize_audio_from_text(transcription_text, synthesized_audio_output)
#PROVIDES MANY OUTPUTS
compare_audio_files(original_audio_input, synthesized_audio_file)


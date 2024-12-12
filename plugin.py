import whisper
import parselmouth
from parselmouth.praat import call
import subprocess
import re 
from transformers import pipeline
# from TTS.api import TTS
import wave
from pydub import AudioSegment
import librosa
import numpy as np
from scipy.spatial.distance import cdist
import cv2
import mediapipe as mp
import time

from collections import Counter

model = whisper.load_model("small")
corrector = pipeline("text2text-generation", model="pszemraj/flan-t5-large-grammar-synthesis")

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

filler_words = ["uh", "um", "like", "you know", "er", "ah", "so", "actually", "basically"]

pattern = r'\b(' + '|'.join(filler_words) + r')\b'

#INPUT FILE IS AUDIO.M4A. audio.wav should NOT be changed.
# subprocess.run(["ffmpeg", "-i", "audio.m4a", "audio.wav"])

def give_transcript(audio_path):
    result = model.transcribe(audio_path)
    transcription = result['text']
    return transcription

def lines_clines(transcript):
    lines = transcript.strip().split('\n')

    corrected_lines = []
    for i, line in enumerate(lines):
        corrected_text = corrector(line)[0]['generated_text']
        corrected_lines.append(corrected_text)
    return lines, corrected_lines

def  corrected_ts(transcript):
    return "\n".join(lines_clines(transcript)[1])

def check_errors(transcript):
    correct_lines, incorrect_lines = lines_clines(transcript)
    d = {}
    for i, (correct, incorrect) in enumerate(zip(correct_lines, incorrect_lines)):
        if correct != incorrect:
            d[i+1] = [correct, incorrect]
    return d


def dets(audio_path, transcript):
    sound = parselmouth.Sound(audio_path)
    duration = call(sound, "Get total duration")
    silences = call(sound, "To TextGrid (silences)", 100, 0.1, -25, 0.1, 0.05, "silent", "sounding")
    n_pauses = call(silences, "Count intervals where", 1, "is equal to", "silent")
    n_words = len(transcript.split())
    speaking_rate = round(n_words / duration, 2 )
    matches = re.findall(pattern, transcript, flags=re.IGNORECASE)
    word_counts = Counter(matches)
    
    return {"duration": duration, "pauses": n_pauses, "rate": speaking_rate, "filler": matches, "filler_count": word_counts}





class BodyLanguageScorer:
    def __init__(self):
        self.scores = {"posture": 0, "gestures": 0, "eye_contact": 100}
        self.gesture_count = 0
        self.eye_movement_threshold = 0.05

    def analyze(self, pose_landmarks, hand_landmarks, face_landmarks):
        if pose_landmarks:
            left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            self.scores["posture"] = max(0, 1 - shoulder_diff) * 100

        if hand_landmarks:
            self.gesture_count += len(hand_landmarks)
            self.scores["gestures"] = self.gesture_count * 25

        if face_landmarks:
            left_eye = face_landmarks[0].landmark[33]
            right_eye = face_landmarks[0].landmark[133]
            eye_distance = np.linalg.norm([left_eye.x - right_eye.x, left_eye.y - right_eye.y])
            if eye_distance > self.eye_movement_threshold:
                self.scores["eye_contact"] = max(0, 100 - eye_distance * 100)

    def get_scores(self):
        return self.scores



def process_video_optimized(video_path):
    cap = cv2.VideoCapture(video_path)
    scorer = BodyLanguageScorer()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, total_frames // 200)  # Process ~200 frames max



    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        for frame_idx in range(0, total_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret :  
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            pose_results = pose.process(image)
            hands_results = hands.process(image)
            face_results = face_mesh.process(image)

            scorer.analyze(
                pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None,
                hands_results.multi_hand_landmarks,
                face_results.multi_face_landmarks
            )

    cap.release()

    scores = scorer.get_scores()
    fn = ""
    if scores["posture"] < 60 or scores["eye_contact"] < 70:
        fn += "Warning: Poor posture or eye contact detected! <br>"

    #OUTPUTS THE FINAL SCORES
    fn += f"""Final Scores: <br>           Posture: {scores['posture']:.2f} <br>            Gestures: {scores['gestures']:.2f} <br>            Eye Contact: {scores['eye_contact']:.2f}"""
    return fn





# tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DCA")

# def synthesize_audio_from_text(text_input, output_file):
#     # Generate synthesized audio using FastSpeech2
#     tts_model.tts_to_file(text=text_input, file_path=output_file)
#     return output_file

# def compare_audio_files(original_audio_file, synthesized_audio_file):
#     original_audio, sr_original = librosa.load(original_audio_file, sr=None)
#     synthesized_audio, sr_synthesized = librosa.load(synthesized_audio_file, sr=None)

#     mfcc_original = librosa.feature.mfcc(y=original_audio, sr=sr_original, n_mfcc=13)
#     mfcc_synthesized = librosa.feature.mfcc(y=synthesized_audio, sr=sr_synthesized, n_mfcc=13)

#     distance = cdist(mfcc_original.T, mfcc_synthesized.T, 'cosine')

#     print(f"Average Cosine Distance between Original and Synthesized MFCCs: {np.mean(distance)}")
#     if np.mean(distance) > 0.5:  
#         print("Pronunciation error detected!")
#     else:
#         print("Pronunciation is correct.")

#     speech_rate_original = len(original_audio) / sr_original
#     speech_rate_synthesized = len(synthesized_audio) / sr_synthesized
#     print(f"Original Speech Rate: {speech_rate_original:.2f} sec")
#     print(f"Synthesized Speech Rate: {speech_rate_synthesized:.2f} sec")

#     pitch_original = librosa.yin(original_audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
#     pitch_synthesized = librosa.yin(synthesized_audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))

#     pitch_variation_original = np.std(pitch_original)
#     pitch_variation_synthesized = np.std(pitch_synthesized)

#     print(f"Pitch Variation Original: {pitch_variation_original:.2f}")
#     print(f"Pitch Variation Synthesized: {pitch_variation_synthesized:.2f}")


# transcription_text = transcription
# original_audio_input = "audio.wav" #PATH FOR ORIGINAL AUDIO IN .WAV FORMAT
# synthesized_audio_output = "synthesized_audio.wav"  #PATH FOR SYNTHESIZED AUDIO
#IT CREATES A NEW FILE
# synthesized_audio_file = synthesize_audio_from_text(transcription_text, synthesized_audio_output)
# #PROVIDES MANY OUTPUTS
# compare_audio_files(original_audio_input, synthesized_audio_file)


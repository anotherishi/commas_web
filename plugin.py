import whisper
import parselmouth
from parselmouth.praat import call
import re
from transformers import pipeline
from pydub import AudioSegment
import librosa
import numpy as np
from scipy.spatial.distance import cdist
import cv2
import mediapipe as mp
import requests
import spacy
import time

from collections import Counter

model = whisper.load_model("small")
corrector = pipeline(
    "text2text-generation", model="pszemraj/flan-t5-large-grammar-synthesis"
)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

filler_words = [
    "uh",
    "um",
    "like",
    "you know",
    "er",
    "ah",
    "so",
    "actually",
    "basically",
    "umm",
    "hmm",
    "uhhhh",
    "huuuu",
    "ahhhh",
    "hhhh",
    "shhhh",
    "dhhhh",
    "like",
    "you know",
    "actually",
    "basically",
    "so",
    "I mean",
    "sort of",
    "kind of",
    "right",
    "just",
    "huh",
    "okay",
    "then",
    "actually",
    "and all",
    "you see",
    "really",
    "anyways",
    "the thing is",
    "umm",
    "hmm",
    "uhhhh",
    "huuuu",
    "ahhhh",
    "hhhh",
    "shhhh",
    "dhhhh",
    "okay",
    "huh",
    "mmm",
    "toh",
    "aaaah",
    "ooh",
    "mmm hmm",
    "uh huh",
    "hmm okay",
    "hmm yeah",
    "yeah",
    "uhm",
    "ahh",
    "eh",
    "ummm",
    "huhh",
    "oh",
    "ohh",
    "ah",
    "huh okay",
    "umm yeah",
    "right",
    "well",
    "so yeah",
    "and um",
    "kind of",
    "sort of",
    "you see",
    "like yeah",
    "basically yeah",
    "uh yeah",
    "actually um",
    "umm right",
    "well um",
    "yeah uh",
    "umm okay",
    "ehh",
    "oh okay",
    "oh um",
    "yeah so",
    "so um",
    "ah um",
    "uh okay",
    "uh well",
    "you know yeah",
    "hmm uh",
    "hmm uh yeah",
    "actually uh",
    "so right",
    "uhmm",
    "err",
    "ahhh um",
    "hhhhh",
]

pattern = r"\b(" + "|".join(filler_words) + r")\b"

# INPUT FILE IS AUDIO.M4A. audio.wav should NOT be changed.
# subprocess.run(["ffmpeg", "-i", "audio.m4a", "audio.wav"])


nlp = spacy.load("en_core_web_sm")
url = "https://api.languagetool.org/v2/check"


import os
import librosa
import numpy as np
from scipy.spatial.distance import cdist


def synthesize_audio_from_text(text_input, output_file, gender):
    if gender == "female":
        os.system(
            f'edge-tts --text "{text_input}" --write-media "{output_file}" --voice "en-IN-NeerjaNeural"'
        )
    if gender == "male":
        os.system(
            f'edge-tts --text "{text_input}" --write-media "{output_file}" --voice "en-IN-PrabhatNeural"'
        )
    return output_file


def compare_audio_files(original_audio_file, synthesized_audio_file):
    original_audio, sr_original = librosa.load(original_audio_file, sr=None)
    synthesized_audio, sr_synthesized = librosa.load(synthesized_audio_file, sr=None)

    mfcc_original = librosa.feature.mfcc(y=original_audio, sr=sr_original, n_mfcc=13)
    mfcc_synthesized = librosa.feature.mfcc(
        y=synthesized_audio, sr=sr_synthesized, n_mfcc=13
    )

    distance = cdist(mfcc_original.T, mfcc_synthesized.T, "cosine")
    avg_distance = np.mean(distance)

    # Evaluate pronunciation accuracy
    print(
        f"Average Cosine Distance between Original and Synthesized MFCCs: {avg_distance}"
    )

    speech_rate_original = len(original_audio) / sr_original
    speech_rate_synthesized = len(synthesized_audio) / sr_synthesized

    pitch_original = librosa.yin(
        original_audio, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C8")
    )
    pitch_synthesized = librosa.yin(
        synthesized_audio, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C8")
    )

    # Evaluate pitch variation
    pitch_variation_original = np.std(pitch_original)
    pitch_variation_synthesized = np.std(pitch_synthesized)

    dit = {
        "avg_distance": avg_distance,
        "speech_rate_original": speech_rate_original,
        "speech_rate_synthesized": speech_rate_synthesized,
        "pitch_variation_original": pitch_variation_original,
        "pitch_variation_synthesized": pitch_variation_synthesized,
    }

    return dit


def comp_pronun(
    transcription, original_audio_input, synthesized_audio_output, gender="female"
):
    # Define the transcription and audio file paths
    transcription_text = transcription

    # Step 1: Synthesize audio from transcription text
    synthesize_audio_from_text(transcription_text, synthesized_audio_output, gender)

    # Step 2: Compare original and synthesized audio
    return compare_audio_files(original_audio_input, synthesized_audio_output)


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def calculate_pronunciation_score(
    dt,
    mfcc_range=(0, 5),
    word_rate_range=(0, 10),
    pitch_var_range=(0, 3500),
    weights=(0.5, 0.3, 0.2),
):
    # Normalize MFCC distance

    mfcc_dist = dt["avg_distance"]
    word_rate_actual = dt["speech_rate_original"]
    word_rate_ideal = dt["speech_rate_synthesized"]
    pitch_var_actual = dt["pitch_variation_original"]
    pitch_var_ideal = dt["pitch_variation_synthesized"]
    mfcc_norm = normalize(mfcc_dist, *mfcc_range)

    # Calculate differences and normalize
    word_rate_diff = abs(word_rate_actual - word_rate_ideal)
    pitch_var_diff = abs(pitch_var_actual - pitch_var_ideal)

    word_rate_score = 1 - normalize(
        word_rate_diff, 0, word_rate_range[1]
    )  # Smaller diff is better
    pitch_var_score = 1 - normalize(
        pitch_var_diff, 0, pitch_var_range[1]
    )  # Smaller diff is better

    # Weighted combination
    pronunciation_score = (
        weights[0] * (1 - mfcc_norm)  # 1 - mfcc_norm since lower distance is better
        + weights[1] * word_rate_score
        + weights[2] * pitch_var_score
    )

    # Scale score to 0-100
    final_score = pronunciation_score * 100
    return {
        "net": abs(round(final_score, 2)),
        "variance": abs(round(1 - mfcc_norm, 2)),
        "word": abs(round(word_rate_score, 2)),
        "pitch": abs(round(pitch_var_score, 2)),
    }


def return_errors(transcript):
    params = {
        "text": transcript,
        "language": "en",
    }
    result = requests.post(url, data=params).json()
    errors = {}
    errors["data"] = []
    for match in result["matches"]:
        errors["data"].append(
            {
                "message": match["message"],
                "corrections": [i["value"] for i in match["replacements"]],
                "context": match["context"]["text"],
            }
        )
    error_count = 0
    for match in result["matches"]:
        if not contains_proper_noun(match, nlp):
            error_count += 1
    errors["n"] = error_count
    print(errors)
    return errors


def contains_proper_noun(match, nlp):
    doc = nlp(match["context"]["text"])
    for token in doc:
        if token.pos_ == "PROPN":
            return True
    return False


def give_transcript(audio_path):
    result = model.transcribe(audio_path)
    transcription = result["text"]
    return transcription


def lines_clines(transcript):
    lines = transcript.strip().split("\n")

    corrected_lines = []
    for i, line in enumerate(lines):
        corrected_text = corrector(line)[0]["generated_text"]
        corrected_lines.append(corrected_text)
    return lines, corrected_lines


def corrected_ts(transcript):
    return "\n".join(lines_clines(transcript)[1])


def check_errors(transcript):
    correct_lines, incorrect_lines = lines_clines(transcript)
    d = {}
    for i, (correct, incorrect) in enumerate(zip(correct_lines, incorrect_lines)):
        if correct != incorrect:
            d[i + 1] = [correct, incorrect]
    return d


def dets(audio_path, transcript):
    sound = parselmouth.Sound(audio_path)
    duration = call(sound, "Get total duration")
    silences = call(
        sound, "To TextGrid (silences)", 100, 0.1, -25, 0.1, 0.05, "silent", "sounding"
    )
    n_pauses = call(silences, "Count intervals where", 1, "is equal to", "silent")
    n_words = len(transcript.split())
    speaking_rate = round(n_words / duration, 2)
    matches = re.findall(pattern, transcript, flags=re.IGNORECASE)
    word_counts = Counter(matches)

    return {
        "duration": duration,
        "pauses": n_pauses,
        "rate": speaking_rate,
        "filler": matches,
        "filler_count": word_counts,
    }


# class BodyLanguageScorer:
#     def __init__(self):
#         self.scores = {"posture": 0, "gestures": 0, "eye_contact": 100}
#         self.gesture_count = 0
#         self.eye_movement_threshold = 0.05

#     def analyze(self, pose_landmarks, hand_landmarks, face_landmarks):
#         if pose_landmarks:
#             left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
#             right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
#             shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
#             self.scores["posture"] = max(0, 1 - shoulder_diff) * 100

#         if hand_landmarks:
#             self.gesture_count += len(hand_landmarks)
#             self.scores["gestures"] = self.gesture_count * 25

#         if face_landmarks:
#             left_eye = face_landmarks[0].landmark[33]
#             right_eye = face_landmarks[0].landmark[133]
#             eye_distance = np.linalg.norm([left_eye.x - right_eye.x, left_eye.y - right_eye.y])
#             if eye_distance > self.eye_movement_threshold:
#                 self.scores["eye_contact"] = max(0, 100 - eye_distance * 100)

#     def get_scores(self):
#         return self.scores


# def process_video_optimized(video_path):
#     cap = cv2.VideoCapture(video_path)
#     scorer = BodyLanguageScorer()

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_skip = max(1, total_frames // 200)  # Process ~200 frames max


#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
#          mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
#          mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

#         for frame_idx in range(0, total_frames, frame_skip):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#             ret, frame = cap.read()
#             if not ret :
#                 break

#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False

#             pose_results = pose.process(image)
#             hands_results = hands.process(image)
#             face_results = face_mesh.process(image)

#             scorer.analyze(
#                 pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None,
#                 hands_results.multi_hand_landmarks,
#                 face_results.multi_face_landmarks
#             )

#     cap.release()

#     scores = scorer.get_scores()
#     res = {}
#     if scores["posture"] < 60 or scores["eye_contact"] < 70:
#         res["warning"] = "Poor posture or eye contact detected!"
#     res["posture"] = f"{scores['posture']:.2f}"
#     res["gestures"] = f"{scores['gestures']:.2f}"
#     res["eye_contact"] = f"{scores['eye_contact']:.2f}"
#     return res


class BodyLanguageScorer:
    def __init__(self):
        self.scores = {"posture": 0, "gestures": 0, "eye_contact": 100}
        self.gesture_count = 0
        self.eye_movement_threshold = 0.05  # Threshold for detecting eye movement

    def analyze_posture(self, pose_landmarks):
        # Example: Calculate shoulder alignment (ideal posture has aligned shoulders)
        left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        self.scores["posture"] = max(0, 1 - shoulder_diff) * 100  # Normalize to a score

    def analyze_gestures(self, hand_landmarks):
        if hand_landmarks:
            # Example: Count visible hands and track movements
            self.gesture_count += len(
                hand_landmarks
            )  # Increase count for visible hands
            self.scores["gestures"] = (
                self.gesture_count * 25
            )  # Example scoring metric for gestures

    def analyze_eye_contact(self, face_landmarks):
        if face_landmarks:
            # Example: Track the eye movement by calculating the distance between eye landmarks
            left_eye = face_landmarks[0].landmark[33]  # Left eye center
            right_eye = face_landmarks[0].landmark[133]  # Right eye center
            eye_distance = np.linalg.norm(
                [left_eye.x - right_eye.x, left_eye.y - right_eye.y]
            )

            if eye_distance > self.eye_movement_threshold:
                self.scores["eye_contact"] = max(
                    0, 100 - eye_distance * 100
                )  # Decrease score with movement

    def get_scores(self):
        return self.scores


# Video processing
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    scorer = BodyLanguageScorer()

    # Set the video frame rate to ensure processing within 20 seconds
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps * 0.1)  # Process every 10th frame (~10% of the total frames)

    start_time = time.time()  # Track processing time

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose, mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands, mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Stop if 20 seconds have passed
            if time.time() - start_time > 20:
                break

            # Skip frames to speed up processing
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
                continue

            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process pose, hands, and face
            pose_results = pose.process(image)
            hands_results = hands.process(image)
            face_results = face_mesh.process(image)

            # Analyze posture
            if pose_results.pose_landmarks:
                scorer.analyze_posture(pose_results.pose_landmarks.landmark)

            # Analyze gestures
            if hands_results.multi_hand_landmarks:
                scorer.analyze_gestures(hands_results.multi_hand_landmarks)

            # Analyze eye contact
            if face_results.multi_face_landmarks:
                scorer.analyze_eye_contact(face_results.multi_face_landmarks)

    cap.release()

    # Get and print final scores
    scores = scorer.get_scores()
    res = {}
    if scores["posture"] < 60 or scores["eye_contact"] < 70:
        res["warning"] = "Poor posture or eye contact detected!"
    res["posture"] = f"{scores['posture']:.2f}"
    res["gestures"] = f"{scores['gestures']:.2f}"
    res["eye_contact"] = f"{scores['eye_contact']:.2f}"
    return res


def calculate_accuracy_score(
    pronunciation_score, error_count, speaking_rate, pause_count, filler_word_count
):
    """
    Calculate accuracy score out of 100 based on industrial standards.

    Parameters:
        pronunciation_score (float): Pronunciation score in percentage (0-100).
        error_count (int): Number of errors.
        speaking_rate (float): Speaking rate in words per second.
        pause_count (int): Number of pauses.
        filler_word_count (int): Number of filler words.

    Returns:
        float: Accuracy score out of 100.
    """
    # Weighting factors (adjust based on importance of each parameter)
    weights = {
        "pronunciation": 0.4,
        "error": 0.2,
        "speaking_rate": 0.2,
        "pause": 0.1,
        "filler_word": 0.1,
    }

    # Normalize each parameter to a 0-1 scale
    normalized_pronunciation = pronunciation_score / 100  # Already percentage

    # Error count: More errors = lower score (assuming a max reasonable error threshold of 20)
    normalized_error = max(0, 1 - error_count / 20)

    # Speaking rate: Optimal range assumed to be 2-4 words/sec
    if 2 <= speaking_rate <= 4:
        normalized_speaking_rate = 1  # Ideal rate
    else:
        normalized_speaking_rate = max(0, 1 - abs(speaking_rate - 3) / 3)

    # Pause count: More pauses = lower score (assuming max reasonable pauses = 10)
    normalized_pause = max(0, 1 - pause_count / 10)

    # Filler word count: More fillers = lower score (assuming max reasonable fillers = 10)
    normalized_filler = max(0, 1 - filler_word_count / 10)

    # Weighted sum of normalized scores
    score = (
        weights["pronunciation"] * normalized_pronunciation
        + weights["error"] * normalized_error
        + weights["speaking_rate"] * normalized_speaking_rate
        + weights["pause"] * normalized_pause
        + weights["filler_word"] * normalized_filler
    )

    # Convert to a percentage score out of 100
    return round(score * 100, 2)


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
# IT CREATES A NEW FILE
# synthesized_audio_file = synthesize_audio_from_text(transcription_text, synthesized_audio_output)
# #PROVIDES MANY OUTPUTS
# compare_audio_files(original_audio_input, synthesized_audio_file)

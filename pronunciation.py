# import os
# import librosa
# import numpy as np
# from scipy.spatial.distance import cdist

# def synthesize_audio_from_text(text_input, output_file, gender):
#     if gender == 'female' :
#         os.system(f'edge-tts --text "{text_input}" --write-media "{output_file}" --voice "en-IN-NeerjaNeural"')
#     if gender == 'male' : 
#         os.system(f'edge-tts --text "{text_input}" --write-media "{output_file}" --voice "en-IN-PrabhatNeural"')       
#     return output_file

# def compare_audio_files(original_audio_file, synthesized_audio_file):
#     original_audio, sr_original = librosa.load(original_audio_file, sr=None)
#     synthesized_audio, sr_synthesized = librosa.load(synthesized_audio_file, sr=None)

#     mfcc_original = librosa.feature.mfcc(y=original_audio, sr=sr_original, n_mfcc=13)
#     mfcc_synthesized = librosa.feature.mfcc(y=synthesized_audio, sr=sr_synthesized, n_mfcc=13)

#     distance = cdist(mfcc_original.T, mfcc_synthesized.T, 'cosine')
#     avg_distance = np.mean(distance)

#     # Evaluate pronunciation accuracy
#     print(f"Average Cosine Distance between Original and Synthesized MFCCs: {avg_distance}")

#     speech_rate_original = len(original_audio) / sr_original
#     speech_rate_synthesized = len(synthesized_audio) / sr_synthesized

#     pitch_original = librosa.yin(original_audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
#     pitch_synthesized = librosa.yin(synthesized_audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))

#     # Evaluate pitch variation
#     pitch_variation_original = np.std(pitch_original)
#     pitch_variation_synthesized = np.std(pitch_synthesized)

#     dit = {
#         "avg_distance" : avg_distance,
#         "speech_rate_original" : speech_rate_original,
#         "speech_rate_synthesized" : speech_rate_synthesized, 
#         "pitch_variation_original" : pitch_variation_original, 
#         "pitch_variation_synthesized" : pitch_variation_synthesized
#     }

#     return dit

# def comp_pronun(transcription,original_audio_input, synthesized_audio_output, gender='female' ):
#     # Define the transcription and audio file paths
#     transcription_text = transcription

#     # Step 1: Synthesize audio from transcription text
#     synthesize_audio_from_text(transcription_text, synthesized_audio_output, gender)

#     # Step 2: Compare original and synthesized audio
#     return compare_audio_files(original_audio_input, synthesized_audio_output)

    

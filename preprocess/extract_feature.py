import numpy as np
import os
import librosa
from transformers import Wav2Vec2FeatureExtractor

def load_audio_files_and_labels(fake_folder, real_folder):
    audio_data = []
    labels = []
    mp3_cnt_fake = 0; mp3_cnt_real = 0
    max_length = -1
    # Load fake audio files
    print("[INFO] total # fake samples:", len(os.listdir(fake_folder)))
    for filename in os.listdir(fake_folder):
        file_path = os.path.join(fake_folder, filename)
        if file_path.endswith(".mp3"):
            mp3_cnt_fake+=1
            continue
        else: audio, sr = librosa.load(file_path, sr=16000, duration=5) 
        if audio.shape[0] > max_length: max_length = audio.shape[0]
        audio_data.append(audio) # (33984,)
        labels.append(1)
    print("[INFO] fake # of mp3:", mp3_cnt_fake)
    
    # Load real audio files
    print("[INFO] total # real samples:", len(os.listdir(real_folder)))
    for filename in os.listdir(real_folder):
        file_path = os.path.join(real_folder, filename)
        
        if file_path.endswith(".mp3"):
            mp3_cnt_real+=1
            continue
        else: audio, sr = librosa.load(file_path, sr=16000, duration=5)  
        if audio.shape[0] > max_length: max_length = audio.shape[0]
        audio_data.append(audio)
        labels.append(0)
    print("[INFO] real # of mp3:", mp3_cnt_real)
    print("[INFO] total # of mp3:", mp3_cnt_real+mp3_cnt_fake)
    print("[INFO] max_length:", max_length)
    return audio_data, np.array(labels), max_length

def padding(x, num_features):
    return np.pad(x[:num_features], (0, num_features - len(x[:num_features])), 'constant', constant_values=0)
    
def extract_features(audio_data):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    features = []
    max_length = -1
    err_idx = []
    for i, audio in enumerate(audio_data):
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="np", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
        feature = inputs.input_values.reshape(-1).tolist()
        if len(feature) > max_length: max_length = len(feature)
        features.append(feature)
    return features, max_length

if __name__ == '__main__':
    fake_root = "/DETECT/fake"
    real_root = "/DETECT/real"

    audio_data, labels, max_length = load_audio_files_and_labels(fake_root, real_root)
    print("[INFO] Audio loaded")
    audio_data_padded = np.array([padding(x, max_length) for x in audio_data])
    print(audio_data_padded.shape)
    np.savez(os.path.join("/DETECT/", f"raw{max_length}.npz"), data=audio_data_padded, labels=labels)
    print("[INFO] Audio padded with", max_length)

    features, max_length = extract_features(audio_data)
    print("[INFO] Feature extracted")
    feature_padded = np.array([padding(x, max_length) for x in features])
    print(feature_padded.shape)
    print("[INFO] Feature padded")
    np.savez(os.path.join("/DETECT/", f"w2v_emb{max_length}.npz"), data=feature_padded, labels=labels)
    print("[INFO] Feature npz saved")
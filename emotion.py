import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

available_emotions = {'angry', 'sad', 'neutral', 'happy', 'fearful', 'disgust', 'calm'}
def augment_audio(audio, sr):
    if np.random.rand() > 0.5:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.randint(-2, 2))
    if np.random.rand() > 0.5:
        rate = np.random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)
    return audio
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        if np.random.rand() > 0.5:
            audio = augment_audio(audio, sr)

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio)

        combined_features = np.hstack((
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(contrast.T, axis=0),
            np.mean(zcr)
        ))
        return combined_features

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
def load_dataset(data_path):
    features = []
    labels = []

    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(subdir, file)
                emotion_code = file.split("-")[2]
                emotion = emotions[emotion_code]

                if emotion not in available_emotions:
                    continue

                feats = extract_features(file_path)
                if feats is not None:
                    features.append(feats)
                    labels.append(emotion)

    return np.array(features), np.array(labels)
#Prepare Data
DATA_PATH = "data/" 
X, y = load_dataset(DATA_PATH)
# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y)

# Reshape input for CNN-LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights_dict = dict(enumerate(class_weights))

#cnn lstm model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(available_emotions), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Save model and label encoder
os.makedirs("models", exist_ok=True)
model.save("models/speech_emotion_recognition_model.keras")
joblib.dump(le, "label_encoder.pkl")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Error"
    features = features.reshape(1, 1, -1)
    prediction = model.predict(features)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

print(predict_emotion("path/to/audio.wav"))
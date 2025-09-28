import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

LOAD_PATH = "alphabet_landmarks_2hands.pkl"
MODEL_PATH = "alphabet_model.pkl"
LABEL_PATH = "alphabet_label_encoder.pkl"

# ---- Load Data ----
with open(LOAD_PATH, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# ---- Encode Labels ----
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# ---- Train Model ----
clf = RandomForestClassifier(n_estimators=300, max_depth=18, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train_enc)

# ---- Evaluate ----
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test_enc, y_pred)
print("Test Accuracy: {:.3f}".format(acc))
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# ---- Save Model & Encoder ----
joblib.dump(clf, MODEL_PATH)
joblib.dump(le, LABEL_PATH)
print("Model & label encoder saved.")

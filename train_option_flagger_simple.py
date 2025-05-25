# train_option_flagger_simple.py
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import joblib
from feature_extractor import load_data_and_extract_features # If in a separate file

# --- Configuration ---
OKAY_DIR = "./option_box_samples/okay/"    # Update with your path
FLAGGED_DIR = "./option_box_samples/flagged/" # Update with your path
MODEL_SAVE_PATH = "option_flagger_simple_model.pkl"
SCALER_SAVE_PATH = "option_flagger_simple_scaler.pkl"

# (Include the parse_background_count_from_filename and extract_features_low_complexity, load_data_and_extract_features functions here if not imported)

def main():
    X, y = load_data_and_extract_features(OKAY_DIR, FLAGGED_DIR)

    if X.shape[0] == 0:
        print("No data loaded. Exiting.")
        return

    print(f"Total samples: {X.shape[0]}, Number of features: {X.shape[1]}")
    print(f"Class distribution: Okay (0): {np.sum(y==0)}, Flagged (1): {np.sum(y==1)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_model_name = None
    best_accuracy = 0.0
    best_model_instance = None

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"ROC AUC Score: {auc:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model_instance = model

    if best_model_instance:
        print(f"\n--- Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f} ---")
        joblib.dump(best_model_instance, MODEL_SAVE_PATH)
        joblib.dump(scaler, SCALER_SAVE_PATH)
        print(f"Saved best model to {MODEL_SAVE_PATH} and scaler to {SCALER_SAVE_PATH}")
    else:
        print("Could not determine a best model.")

if __name__ == "__main__":
    # Create dummy directories and files for testing if they don't exist
    # In a real scenario, these should be populated by the user.
    os.makedirs(OKAY_DIR, exist_ok=True)
    os.makedirs(FLAGGED_DIR, exist_ok=True)
    # Example: Create a dummy okay image file if one doesn't exist for testing
    if not os.listdir(OKAY_DIR) and not os.listdir(FLAGGED_DIR) :
        print(f"Please populate {OKAY_DIR} and {FLAGGED_DIR} with sample images named like 'name_count_123.png'")
        # You might want to create a tiny placeholder image for testing script flow:
        # placeholder_img = np.zeros((50, 50, 3), dtype=np.uint8)
        # cv2.imwrite(os.path.join(OKAY_DIR, "sample_okay_count_3000.png"), placeholder_img)
        # cv2.imwrite(os.path.join(FLAGGED_DIR, "sample_flagged_count_1000.png"), placeholder_img)
    main()
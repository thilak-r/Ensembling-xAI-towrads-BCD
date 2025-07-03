# -*- coding: utf-8 -*-
"""
Flask App for Breast Cancer Prediction with Grad-CAM Visualization Pipeline.
Uses a YOLOv8 detector + EfficientNetB0 classifier trained on masked data.
"""

import os
import io
import base64
from PIL import Image as PILImage
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models as torchvision_models
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import torch.nn as nn

# Import the functions from your pipeline module
# Make sure utils/gradcam.py is in your Python path
try:
    from utils.gradcam import load_pipeline_models, generate_gradcam
except ImportError:
    print("Error: Could not import 'utils.gradcam'. Ensure utils/gradcam.py exists and is in your Python path.")
    # Define dummy functions to prevent crashes if the module is missing
    def load_pipeline_models(*args, **kwargs): print("Dummy load_pipeline_models called."); return False
    # Updated dummy generate_gradcam to return 5 values
    def generate_gradcam(*args, **kwargs): print("Dummy generate_gradcam called."); return None, None, 0.0, "Module Not Loaded", 0.0


# Import your other models if needed (Stage 1/Stage 2 multimodal)
# Adjust paths as needed based on your project structure
try:
    from utils.models import CNN_Model, get_torchvision_model, MultimodalBreastCancerCNN
except ImportError:
    print("Warning: Could not import 'utils.models'. Stage 1/Stage 2 models might not load.")
    # Define dummy functions/classes if necessary
    class CNN_Model(nn.Module):
        def __init__(self, num_classes=1): super().__init__(); self.fc = nn.Identity()
        def forward(self, x): return torch.zeros(x.size(0), num_classes)
    def get_torchvision_model(name, num_classes=1):
        print(f"Warning: Dummy model '{name}' loaded.")
        class DummyModel(nn.Module):
             def __init__(self): super().__init__(); self.fc = nn.Identity()
             def forward(self, x): return torch.zeros(x.size(0), num_classes)
        return DummyModel()
    class MultimodalBreastCancerCNN(nn.Module):
         def __init__(self, num_clinical_features, num_classes=1): super().__init__(); self.fc = nn.Identity()
         def forward(self, img, clin): return torch.zeros(img.size(0), num_classes), None


import cv2

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_super_secret_key_meow_v2' # CHANGE THIS IN PRODUCTION
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Flask App: Using device: {DEVICE}")

# --- Model Paths (for Stage 1/Stage 2 - Adjust these as needed) ---
MODELS_DIR = 'models/'
stage1_model_paths = {
    "cnn": os.path.join(MODELS_DIR, "best_kappa_model_cnn.pth"),
    "resnet18": os.path.join(MODELS_DIR, "best_kappa_model_resnet18.pth"),
    "densenet121": os.path.join(MODELS_DIR, "best_kappa_model_densenet121.pth"),
    "resnet50": os.path.join(MODELS_DIR, "best_kappa_model_resnet50.pth"),
    "vgg16": os.path.join(MODELS_DIR, "best_kappa_model_vgg16.pth"),
}
stage2_multimodal_model_path = os.path.join(MODELS_DIR, "best_multimodal_model.pth")

# Paths for the pipeline models (YOLO Detector + EfficientNetB0 Classifier)
PIPELINE_CLASSIFIER_WEIGHT_PATH = os.path.join(MODELS_DIR, 'efficientnet_b0_best_auc.pth')
PIPELINE_DETECTOR_WEIGHT_PATH = os.path.join(MODELS_DIR, 'yolov8_breast_detector_yolov8_best.pt')


# --- Clinical Data Scaling Parameters ---
CLINICAL_MEANS = np.array([48.714285714285715, 72.26470588235294, 157.05714285714285, 35.99915966386554])
CLINICAL_STDS = np.array([13.102147400582952, 16.884768076012232, 7.137068242439532, 0.4296756248318925])
NUM_CLINICAL_FEATURES = 4

# --- Image Transformations (Define here) ---
transform_stage1 = transforms.Compose([
    transforms.Resize((120, 160)), # Example size for Stage 1 models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_stage2_multimodal_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- Load Models (Global Variables) ---
stage1_models = {}
stage2_multimodal_model = None

def load_all_models():
    """Loads all models needed by the Flask app."""
    global stage1_models, stage2_multimodal_model
    
    print("\n--- Loading Models for Flask App ---")

    # Load Stage 1 ensemble models
    print("Loading Stage 1 models...")
    try:
        cnn_model_instance = CNN_Model(num_classes=1).to(DEVICE)
        if os.path.exists(stage1_model_paths["cnn"]):
            try: cnn_model_instance.load_state_dict(torch.load(stage1_model_paths["cnn"], map_location=DEVICE, weights_only=True))
            except TypeError: cnn_model_instance.load_state_dict(torch.load(stage1_model_paths["cnn"], map_location=DEVICE))
            except Exception as e: print(f"Error loading state_dict for custom CNN: {e}"); cnn_model_instance = None
            if cnn_model_instance: cnn_model_instance.eval(); stage1_models["cnn"] = cnn_model_instance; print("Custom CNN model loaded.")
        else: print(f"ERROR: Custom CNN model file not found at {stage1_model_paths['cnn']}")
    except Exception as e: print(f"Error loading custom CNN model: {e}")

    for name in ["resnet18", "densenet121", "resnet50", "vgg16"]:
        try:
            model_instance = get_torchvision_model(name, num_classes=1).to(DEVICE)
            if os.path.exists(stage1_model_paths[name]):
                 try: model_instance.load_state_dict(torch.load(stage1_model_paths[name], map_location=DEVICE, weights_only=True))
                 except TypeError: model_instance.load_state_dict(torch.load(stage1_model_paths[name], map_location=DEVICE))
                 except Exception as e: print(f"Error loading state_dict for Stage 1 {name}: {e}"); model_instance = None
                 if model_instance: model_instance.eval(); stage1_models[name] = model_instance; print(f"Stage 1 {name} model loaded.")
            else: print(f"ERROR: Stage 1 {name} model file not found at {stage1_model_paths[name]}")
        except Exception as e: print(f"Error loading Stage 1 {name} model: {e}")

    print("\nLoading Stage 2 (Multimodal) model...")
    try:
        stage2_multimodal_model = MultimodalBreastCancerCNN(num_clinical_features=NUM_CLINICAL_FEATURES, num_classes=1).to(DEVICE)
        if os.path.exists(stage2_multimodal_model_path):
             try: stage2_multimodal_model.load_state_dict(torch.load(stage2_multimodal_model_path, map_location=DEVICE, weights_only=True))
             except TypeError: stage2_multimodal_model.load_state_dict(torch.load(stage2_multimodal_model_path, map_location=DEVICE))
             except Exception as e: print(f"Error loading state_dict for Multimodal: {e}"); stage2_multimodal_model = None
             if stage2_multimodal_model: stage2_multimodal_model.eval(); print("Multimodal model loaded.")
        else: print(f"ERROR: Multimodal model file not found at {stage2_multimodal_model_path}"); stage2_multimodal_model = None
    except Exception as e: print(f"Error loading Stage 2 model: {e}"); stage2_multimodal_model = None

    # Load the Pipeline Models (Detector + EfficientNetB0 Classifier)
    print("\nLoading Pipeline Models (Detector + EfficientNetB0 Classifier)...")
    # load_pipeline_models will update the global variables inside utils.gradcam
    pipeline_models_loaded = load_pipeline_models(
        classifier_name='efficientnet_b0',
        classifier_weights_path=PIPELINE_CLASSIFIER_WEIGHT_PATH,
        detector_weights_path=PIPELINE_DETECTOR_WEIGHT_PATH
    )
    if pipeline_models_loaded: print("Pipeline models loaded successfully.")
    else: print("Warning: Pipeline models failed to load. Grad-CAM visualization will not be available.")


# Call the loading function when the app starts
# Note: In debug mode (debug=True), Flask might restart and call this twice.
# The load_pipeline_models function in utils/gradcam is designed to handle this.
load_all_models()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_clinical_data(age, weight, height, temp):
    try:
        raw_features = np.array([float(age), float(weight), float(height), float(temp)], dtype=np.float32)
    except ValueError: return None
    scaled_features = (raw_features - CLINICAL_MEANS) / CLINICAL_STDS
    return torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('No image file part'); return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        flash('No image selected'); return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try: file.save(img_path)
        except Exception as e: flash(f"Error saving file: {str(e)}"); return redirect(request.url)

        # Read the image as NumPy array (RGB) for pipeline and potentially Stage 1/2
        try:
            img_np = cv2.imread(img_path)
            if img_np is None:
                 flash(f"Error loading image from {img_path}"); return redirect(request.url)
            if len(img_np.shape) == 2: img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else: img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        except Exception as e:
            flash(f"Error processing image with OpenCV: {str(e)}"); return redirect(request.url)

        # Convert NumPy RGB to PIL Image for transforms that expect PIL
        img_pil = PILImage.fromarray(img_np_rgb)

        try:
            age_str, weight_str, height_str, temp_str = request.form['age'], request.form['weight'], request.form['height'], request.form['temp']
        except KeyError: flash("Missing clinical data fields."); return redirect(request.url)


        stage1_results, overall_sick_votes, sick_confidences = [], 0, []
        img_tensor_s1 = transform_stage1(img_pil).unsqueeze(0).to(DEVICE)

        for model_name, model in stage1_models.items():
            if model is None:
                stage1_results.append({"name": model_name, "error": "Not loaded"})
                continue
            with torch.no_grad():
                logit = model(img_tensor_s1)
                prob  = torch.sigmoid(logit).item()

            label = "Sick" if prob > 0.5 else "Normal"
            conf  = prob if label == "Sick" else 1 - prob
            stage1_results.append({"name": model_name,
                                "prediction": label,
                                "confidence": f"{conf*100:.2f}%"})

            if label == "Sick":
                overall_sick_votes += 1
                sick_confidences.append(prob)

        is_overall_sick      = overall_sick_votes >= len(stage1_models) / 2
        avg_sick_confidence  = (np.mean(sick_confidences)
                                if sick_confidences else 0.0)

        # =================== BRANCH HERE ==========================
        if not is_overall_sick:
            # Stage 1 says “Normal”: skip the rest.
            return render_template(
                'index.html',
                filename=filename,
                stage1_results=stage1_results,
                is_overall_sick=False,
                avg_sick_confidence="N/A",
                stage2_prediction="Skipped",
                stage2_confidence="N/A",
                pipeline_prediction="Skipped",
                pipeline_confidence="N/A",
                grad_cam_image=None,
                detected_bbox=None,
                detection_confidence="N/A"
            )
        # =========================================================
        else:
            img_gray = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2GRAY)

# 2. Gray ➜ 3‑channel “gray‑RGB” so the CNNs still receive 3 channels
            img_gray_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

# 3. Build a PIL image for torchvision transforms
            img_pil_gray = PILImage.fromarray(img_gray_rgb)

        # ---------- Stage 2 (Multimodal) ----------
        stage2_multimodal_prediction_label, stage2_multimodal_confidence_str = "Model Not Loaded", "N/A"
        if stage2_multimodal_model:                      # Executes only if sick
            clinical_tensor = preprocess_clinical_data(age_str, weight_str, height_str, temp_str)
            if clinical_tensor is not None:
                clinical_tensor = clinical_tensor.to(DEVICE)
                img_tensor_s2   = transform_stage2_multimodal_image(img_pil_gray).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logit, _ = stage2_multimodal_model(img_tensor_s2, clinical_tensor)
                prob = torch.sigmoid(logit).item()
                stage2_multimodal_prediction_label = ("Malignant" if prob > 0.5 else "Benign")
                stage2_multimodal_confidence_str   = f"{(prob if prob>0.5 else 1-prob)*100:.2f}%"

        # ---------- Stage 3 (Grad‑CAM Pipeline) ----------
        grad_cam_img_b64          = None
        pipeline_prediction_label = "Pipeline Skipped"
        pipeline_confidence_score = 0.0
        detected_bbox_coords      = None
        detection_confidence_score = 0.0

        print("\nRunning Grad-CAM Visualization Pipeline...")
        # Call the generate_gradcam function from your module
        # It takes the image NumPy array (RGB) as input
        try:
             # --- Capture all 5 expected return values ---
             cam_overlay_np, detected_box_pixel, detection_confidence_score, pipeline_prediction_label, pipeline_confidence_score = generate_gradcam(
                 img_gray_rgb,
                 detector_conf_threshold=0.5, # Use config value or adjust
                 heatmap_threshold_ratio=0.6 # Use config value or adjust
             )

             if cam_overlay_np is not None:
                 # Convert the resulting NumPy array (uint8 RGB) to PIL and then base64 encode
                 cam_overlay_pil = PILImage.fromarray(cam_overlay_np)
                 buffered = io.BytesIO()
                 cam_overlay_pil.save(buffered, format="PNG")
                 grad_cam_img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                 print("Pipeline Grad-CAM generated and encoded.")
                 detected_bbox_coords = detected_box_pixel # Store the detected box from the pipeline
             else:
                  print("Pipeline Grad-CAM visualization failed or no box detected.")
                  # pipeline_prediction_label and pipeline_confidence_score already reflect the reason from generate_gradcam


        except Exception as e:
             print(f"Error running pipeline (generate_gradcam): {e}")
             flash(f"Error running analysis pipeline: {str(e)}")
             pipeline_prediction_label = "Pipeline Error"
             pipeline_confidence_score = 0.0
             grad_cam_img_b64 = None
             detected_bbox_coords = None
             detection_confidence_score = 0.0


        # --- Render Template ---
        return render_template('index.html',
                               filename=filename,
                               stage1_results=stage1_results,
                               is_overall_sick=is_overall_sick,
                               avg_sick_confidence=f"{avg_sick_confidence*100:.2f}%" if avg_sick_confidence is not None else "N/A",
                               stage2_prediction=stage2_multimodal_prediction_label,
                               stage2_confidence=stage2_multimodal_confidence_str,

                               pipeline_prediction=pipeline_prediction_label,
                               pipeline_confidence=f"{pipeline_confidence_score*100:.2f}%" if pipeline_confidence_score is not None else "N/A",
                               grad_cam_image=grad_cam_img_b64,
                               detected_bbox=detected_bbox_coords, # Pass detected box [x_min, y_min, x_max, y_max] or None
                               detection_confidence=f"{detection_confidence_score*100:.2f}%" if detection_confidence_score is not None and detection_confidence_score > 0 else "N/A"
                               )

    else:
        flash('Allowed image types are png, jpg, jpeg, bmp, tif')
        return redirect(request.url)


if __name__ == '__main__':
    print("\nStarting Flask App...")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
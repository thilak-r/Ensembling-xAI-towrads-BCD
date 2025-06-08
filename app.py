import os
import io
import base64
from PIL import Image as PILImage
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models as torchvision_models # Alias to avoid conflict if models.py also has 'models'
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import torch.nn as nn

# Import your model classes and GradCAM
from utils.models import CNN_Model, get_torchvision_model, MultimodalBreastCancerCNN
# Import the DenseNet121 definition for the Grad-CAM specific model
# Assuming you might have a function like this in utils.models or define one here
try:
    from utils.models import get_densenet121_for_gradcam # If you added it there
except ImportError:
    # Fallback: Define get_densenet121_for_gradcam here if not in utils.models
    def get_densenet121_for_gradcam(num_classes=1, pretrained_weights_path=None): # pretrained_weights_path not used here for loading
        model = torchvision_models.densenet121(weights=None) # Load structure only
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        return model

from utils.gradcam import GradCAM, overlay_gradcam
import cv2

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_super_secret_key_meow_v2'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Flask App: Using device: {DEVICE}")

# --- Model Paths ---
MODELS_DIR = 'models/'
stage1_model_paths = {
    "cnn": os.path.join(MODELS_DIR, "best_kappa_model_cnn.pth"),
    "resnet18": os.path.join(MODELS_DIR, "best_kappa_model_resnet18.pth"),
    "densenet121": os.path.join(MODELS_DIR, "best_kappa_model_densenet121.pth"),
    "resnet50": os.path.join(MODELS_DIR, "best_kappa_model_resnet50.pth"),
    "vgg16": os.path.join(MODELS_DIR, "best_kappa_model_vgg16.pth"),
}
stage2_multimodal_model_path = os.path.join(MODELS_DIR, "best_multimodal_model.pth")
# Path for the separate DenseNet121 trained for Grad-CAM
densenet121_gradcam_viz_model_path = os.path.join(MODELS_DIR, "best_densenet121_for_gradcam.pth")


# --- Clinical Data Scaling Parameters ---
CLINICAL_MEANS = np.array([48.714285714285715, 72.26470588235294, 157.05714285714285, 35.99915966386554])
CLINICAL_STDS = np.array([13.102147400582952, 16.884768076012232, 7.137068242439532, 0.4296756248318925])
NUM_CLINICAL_FEATURES = 4

# --- Load Models (Global Variables) ---
stage1_models = {}
stage2_multimodal_model = None
densenet121_for_gradcam_visualization = None # For the separate Grad-CAM model

def load_all_models():
    global stage1_models, stage2_multimodal_model, densenet121_for_gradcam_visualization
    # ... (Loading for Stage 1 models - CNN_Model, ResNet18, DenseNet121 (ensemble), ResNet50, VGG16 remains the same)
    print("Loading Stage 1 models...")
    try:
        cnn_model_instance = CNN_Model(num_classes=1).to(DEVICE)
        if os.path.exists(stage1_model_paths["cnn"]):
            cnn_model_instance.load_state_dict(torch.load(stage1_model_paths["cnn"], map_location=DEVICE, weights_only=True))
            cnn_model_instance.eval(); stage1_models["cnn"] = cnn_model_instance; print("CNN model loaded.")
        else: print(f"ERROR: CNN model file not found at {stage1_model_paths['cnn']}")
    except Exception as e: print(f"Error loading custom CNN model: {e}")

    for name in ["resnet18", "densenet121", "resnet50", "vgg16"]:
        try:
            model_instance = get_torchvision_model(name, num_classes=1).to(DEVICE)
            if os.path.exists(stage1_model_paths[name]):
                model_instance.load_state_dict(torch.load(stage1_model_paths[name], map_location=DEVICE, weights_only=True))
                model_instance.eval(); stage1_models[name] = model_instance; print(f"{name} model loaded.")
            else: print(f"ERROR: {name} model file not found at {stage1_model_paths[name]}")
        except Exception as e: print(f"Error loading {name} model: {e}")
    
    print("\nLoading Stage 2 (Multimodal) model...")
    try:
        stage2_multimodal_model = MultimodalBreastCancerCNN(num_clinical_features=NUM_CLINICAL_FEATURES, num_classes=1).to(DEVICE)
        if os.path.exists(stage2_multimodal_model_path):
            stage2_multimodal_model.load_state_dict(torch.load(stage2_multimodal_model_path, map_location=DEVICE, weights_only=True))
            stage2_multimodal_model.eval(); print("Multimodal model loaded.")
        else: print(f"ERROR: Multimodal model file not found at {stage2_multimodal_model_path}"); stage2_multimodal_model = None
    except Exception as e: print(f"Error loading Stage 2 model: {e}"); stage2_multimodal_model = None

    print("\nLoading DenseNet121 (for Grad-CAM visualization)...")
    try:
        densenet121_for_gradcam_visualization = get_densenet121_for_gradcam(num_classes=1).to(DEVICE) # num_classes=1 for B/M
        if os.path.exists(densenet121_gradcam_viz_model_path):
            densenet121_for_gradcam_visualization.load_state_dict(torch.load(densenet121_gradcam_viz_model_path, map_location=DEVICE, weights_only=True))
            densenet121_for_gradcam_visualization.eval()
            print("DenseNet121 for Grad-CAM visualization loaded.")
        else:
            print(f"ERROR: DenseNet121 for Grad-CAM model not found at {densenet121_gradcam_viz_model_path}")
            densenet121_for_gradcam_visualization = None
    except Exception as e:
        print(f"Error loading DenseNet121 for Grad-CAM: {e}")
        densenet121_for_gradcam_visualization = None

load_all_models()

# --- Image Transformations ---
transform_stage1 = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_stage2_image_and_gradcam_densenet = transforms.Compose([ # Can be reused for DenseNet121 if input size is same
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

        try:
            age_str, weight_str, height_str, temp_str = request.form['age'], request.form['weight'], request.form['height'], request.form['temp']
        except KeyError: flash("Missing clinical data fields."); return redirect(request.url)

        # --- Stage 1 Prediction ---
        stage1_results, overall_sick_votes, sick_confidences = [], 0, []
        img_pil = PILImage.open(img_path).convert('RGB')
        img_tensor_s1 = transform_stage1(img_pil).unsqueeze(0).to(DEVICE)
        try:
            for model_name, model in stage1_models.items():
                if model is None: stage1_results.append({"name": model_name, "error": "Not loaded"}); continue
                with torch.no_grad():
                    output_s1_logits = model(img_tensor_s1)
                    prob_s1_sick = torch.sigmoid(output_s1_logits).item()
                pred_s1_label = "Sick" if prob_s1_sick > 0.5 else "Normal"
                confidence_s1 = prob_s1_sick if pred_s1_label == "Sick" else 1 - prob_s1_sick
                stage1_results.append({"name": model_name, "prediction": pred_s1_label, "confidence": f"{confidence_s1*100:.2f}%"})
                if pred_s1_label == "Sick": overall_sick_votes += 1; sick_confidences.append(prob_s1_sick)
            is_overall_sick = overall_sick_votes >= (len(stage1_models) / 2.0) # Tie goes to sick if exactly half
            avg_sick_confidence = np.mean(sick_confidences) if sick_confidences else 0.0
        except Exception as e:
            flash(f"Stage 1 Error: {str(e)}"); print(f"Stage 1 Error: {e}")
            return render_template('index.html', filename=filename, error_message=f"Stage 1 Error: {e}")

        # --- Stage 2 (Multimodal Prediction and Separate Grad-CAM) ---
        stage2_multimodal_prediction_label, stage2_multimodal_confidence_str = None, None
        grad_cam_img_b64_densenet = None # For DenseNet121 Grad-CAM

        if is_overall_sick:
            # 1. Get prediction from Multimodal Model
            if stage2_multimodal_model:
                clinical_tensor_s2 = preprocess_clinical_data(age_str, weight_str, height_str, temp_str)
                if clinical_tensor_s2 is None:
                    flash("Invalid clinical data. Stage 2 (Multimodal) skipped.")
                else:
                    clinical_tensor_s2 = clinical_tensor_s2.to(DEVICE)
                    # Use the same PIL image, but transform for stage 2 models (ResNet and DenseNet)
                    img_tensor_s2_or_gradcam = transform_stage2_image_and_gradcam_densenet(img_pil).unsqueeze(0).to(DEVICE)
                    try:
                        stage2_multimodal_model.eval()
                        with torch.no_grad(): # No grad needed for multimodal's own prediction
                            output_s2_logits, _ = stage2_multimodal_model(img_tensor_s2_or_gradcam, clinical_tensor_s2)
                        prob_s2_malignant = torch.sigmoid(output_s2_logits).item()
                        stage2_multimodal_prediction_label = "Malignant" if prob_s2_malignant > 0.5 else "Benign"
                        confidence_s2 = prob_s2_malignant if stage2_multimodal_prediction_label == "Malignant" else 1 - prob_s2_malignant
                        stage2_multimodal_confidence_str = f"{confidence_s2*100:.2f}%"
                    except Exception as e:
                        flash(f"Multimodal Prediction Error: {str(e)}"); print(f"Multimodal Prediction Error: {e}")
                        stage2_multimodal_prediction_label = f"Error processing"
            else:
                flash("Multimodal model (Stage 2) not loaded.")
                stage2_multimodal_prediction_label = "Error: Model not loaded"

            # 2. Generate Grad-CAM using the separate DenseNet121
# ... (inside the predict_bcd route, in the Grad-CAM section for DenseNet121)

            # 2. Generate Grad-CAM using the separate DenseNet121
            if densenet121_for_gradcam_visualization: # This is your loaded model instance
                try:
                    print("Generating Grad-CAM with DenseNet121...")
                    densenet121_for_gradcam_visualization.eval() # Use the correct variable
                    img_tensor_for_densenet_gradcam = transform_stage2_image_and_gradcam_densenet(img_pil).unsqueeze(0).to(DEVICE)
                    img_tensor_for_densenet_gradcam.requires_grad_(True)

                    # Define target layer for DenseNet121
                    # !!! MEOW: Use the correct variable name here !!!
                    target_layer_densenet = densenet121_for_gradcam_visualization.features.denseblock4.denselayer16.conv2
                    # (Remember to verify this layer path from your model printout)
                    
                    print(f"Actual Grad-CAM target for DenseNet121: {type(target_layer_densenet)}")
                    if not isinstance(target_layer_densenet, nn.Module):
                        raise AttributeError(f"Selected target_layer_densenet is not an nn.Module!")

                    # !!! MEOW: Use the correct variable name here !!!
                    grad_cam_hook_densenet = GradCAM(densenet121_for_gradcam_visualization, target_layer_densenet)
                    
                    # Forward pass for DenseNet121
                    # !!! MEOW: Use the correct variable name here !!!
                    densenet_logits_for_gradcam = densenet121_for_gradcam_visualization(img_tensor_for_densenet_gradcam)
                    
                    heatmap_np_densenet = grad_cam_hook_densenet.generate_heatmap(densenet_logits_for_gradcam)

                    if heatmap_np_densenet is not None:
                        original_img_for_gradcam_pil = PILImage.open(img_path).convert('RGB').resize((224,224))
                        original_img_for_gradcam_np = np.array(original_img_for_gradcam_pil)
                        superimposed_img_np_densenet = overlay_gradcam(original_img_for_gradcam_np, heatmap_np_densenet)
                        superimposed_pil_densenet = PILImage.fromarray(superimposed_img_np_densenet)
                        buffered_densenet = io.BytesIO()
                        superimposed_pil_densenet.save(buffered_densenet, format="PNG")
                        grad_cam_img_b64_densenet = base64.b64encode(buffered_densenet.getvalue()).decode('utf-8')
                        print("DenseNet121 Grad-CAM generated.")
                    else:
                        print("DenseNet121 Grad-CAM heatmap generation failed (heatmap_np is None).")
                except AttributeError as ae_gradcam: # Catch issues with finding target_layer_dn
                    flash(f"Grad-CAM Layer Error: {str(ae_gradcam)}. Check DenseNet121 target layer path.")
                    print(f"Grad-CAM AttributeError (target layer): {ae_gradcam}")
                except Exception as e: # General exception for other Grad-CAM errors
                    flash(f"DenseNet121 Grad-CAM Error: {str(e)}")
                    print(f"DenseNet121 Grad-CAM Error: {e}") # Log the full error
            else:
                print("DenseNet121 for Grad-CAM visualization not loaded.")
# ...

        return render_template('index.html',
                               filename=filename,
                               stage1_results=stage1_results,
                               is_overall_sick=is_overall_sick,
                               avg_sick_confidence=f"{avg_sick_confidence*100:.2f}%" if is_overall_sick else None,
                               stage2_prediction=stage2_multimodal_prediction_label, # From multimodal
                               stage2_confidence=stage2_multimodal_confidence_str,   # From multimodal
                               grad_cam_image=grad_cam_img_b64_densenet) # From DenseNet121

    else:
        flash('Allowed image types are png, jpg, jpeg, bmp, tif')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
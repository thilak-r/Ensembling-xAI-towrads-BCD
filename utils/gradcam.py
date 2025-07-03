# -*- coding: utf-8 -*-
"""
Module for breast cancer classification pipeline and Grad-CAM visualization.
Includes object detection for ROI, masking, classification, and CAM generation.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO # Assuming ultralytics is installed

# --- Configuration ---
# Default Model Paths (Can be overridden when calling load_pipeline)
DEFAULT_CLASSIFIER_NAME = 'efficientnet_b0'
DEFAULT_CLASSIFIER_WEIGHT_PATH = r'models/efficientnet_b0_best_auc.pth' # Default local path relative to app.py
DEFAULT_DETECTOR_WEIGHT_PATH = r'models/yolov8_breast_detector_yolov8_best.pt' # Default local path relative to app.py

# Model Input Sizes
CLASSIFIER_IMAGE_SIZE = 224 # Match classifier training size
DETECTOR_IMG_SIZE = 640 # Match detector training size if possible

# Classification Config
CLASSIFIER_CLASS_NAMES = {0: "Benign", 1: "Malignant"}
CLASSIFIER_NUM_CLASSES = len(CLASSIFIER_CLASS_NAMES)

# Detector Config (Used in processing function)
DEFAULT_DETECTOR_CONF_THRESHOLD = 0.5
DEFAULT_DETECTOR_IOU_THRESHOLD = 0.4

# General Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MASK_TYPE = 'zero' # Masking type ('zero' or 'blur')

# Visualization Config (Can be passed to generate_gradcam)
DEFAULT_APPLY_HEATMAP_FILTER = True
DEFAULT_HEATMAP_THRESHOLD_RATIO = 0.6


# --- Global Model Instances ---
classifier_model = None
detector_model = None
classifier_cam_generator = None
_loaded_classifier_name = None
_loaded_detector_path = None


# --- Helper Functions ---

def pixel_to_corners_list(box_pixel, class_id=None):
    """
    Converts a pixel box [x_min, y_min, x_max, y_max] to list format [class_id, x_min, y_min, x_max, y_max].
    """
    x_min, y_min, x_max, y_max = box_pixel
    return [class_id, x_min, y_min, x_max, y_max]


def apply_masking(image, pixel_boxes, mask_type='zero'):
    """
    Applies masking outside the bounding boxes. Assumes a single box per image.
    Pixel box format: [class_id/None, x_min, y_min, x_max, y_max]
    Image is a NumPy array (H, W, C).
    """
    if not pixel_boxes:
         # print("Warning: apply_masking called with no boxes. Returning original image.")
         return image

    class_id, x_min, y_min, x_max, y_max = pixel_boxes[0]

    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max = min(image.shape[1] - 1, x_max)
    y_max = min(image.shape[0] - 1, y_max)

    if x_max <= x_min or y_max <= y_min:
        # print(f"Warning: Invalid bounding box coordinates for masking: [{x_min},{y_min},{x_max},{y_max}]. Returning original image.")
        return image


    masked_image = image.copy()

    if mask_type == 'zero':
        mask = np.zeros_like(image)
        mask[y_min:y_max+1, x_min:x_max+1] = 1
        masked_image = image * mask

    elif mask_type == 'blur':
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y_min:y_max+1, x_min:x_max+1] = 255

        mask_inverse = cv2.bitwise_not(mask)
        blurred_image = cv2.GaussianBlur(image, (51, 51), 0)
        masked_image = np.zeros_like(image)
        for i in range(image.shape[2]):
             masked_image[:,:,i] = np.where(mask == 255, image[:,:,i], blurred_image[:,:,i])

    else:
        print(f"Warning: Unknown mask_type: {mask_type}. Returning original image.")
        return image

    return masked_image


def _load_saved_classifier(model_name, weights_path, num_classes, device):
    """Internal function to load a classification model."""
    print(f"Loading classifier: {model_name} from {weights_path}...")
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        print(f"Error: Unknown classification model name: {model_name}")
        return None

    if not os.path.exists(weights_path):
        print(f"Error: Classifier model weights not found at {weights_path}")
        return None

    try:
        # Use weights_only=True if PyTorch version supports it (>= 2.0)
        # Otherwise, remove weights_only=True
        try:
             state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
             state_dict = torch.load(weights_path, map_location=device)

        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state dict for {model_name} from {weights_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading classifier: {e}")
        return None


    model = model.to(device)
    model.eval()
    print("Classifier loaded successfully.")
    return model


def _get_classifier_target_layer(model_name, model):
    """Internal function to map classifier model name to the target layer for Grad-CAM."""
    if model is None:
         print("Error: Classifier model is None, cannot get target layer.")
         return None

    if model_name == 'resnet18':
        target_layer = model.layer4[-1]
    elif model_name == 'densenet121':
         last_conv = None
         for name, module in reversed(list(model.features.named_modules())):
             if isinstance(module, nn.Conv2d):
                 last_conv = module
                 break
         if last_conv is not None:
              target_layer = last_conv
         else:
              # print(f"Warning: Could not find specific last Conv2d for {model_name}. Using last features module.")
              target_layer = model.features[-1]

    elif model_name == 'efficientnet_b0':
        target_layer = model.features[-1]
    else:
        print(f"Error: No defined target layer for classifier model: {model_name}")
        return None
    return target_layer


# --- Data Transformations for Classifier Input (Moved Here) ---
# These transforms prepare the MASKED image for the classifier
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Define the transform here in the module where it's used
classifier_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
# -----------------------------------------------------


def _filter_heatmap_by_threshold(heatmap, threshold_ratio=0.6):
    """
    Filters a heatmap to keep only values above a certain threshold (relative to max).
    """
    if heatmap is None:
        return None

    heatmap_max = np.max(heatmap)
    if heatmap_max < 1e-6:
        return np.zeros_like(heatmap)

    threshold_value = threshold_ratio * heatmap_max
    binary_mask = (heatmap > threshold_value).astype(np.float32)

    filtered_heatmap = heatmap * binary_mask

    return filtered_heatmap

# --- Model Loading Function (Call This Once) ---
def load_pipeline_models(classifier_name=DEFAULT_CLASSIFIER_NAME, classifier_weights_path=DEFAULT_CLASSIFIER_WEIGHT_PATH, detector_weights_path=DEFAULT_DETECTOR_WEIGHT_PATH):
    """
    Loads the classifier and detector models and initializes the CAM generator.
    This function should be called ONCE before processing images.

    Args:
        classifier_name (str): Name of the classification model architecture.
        classifier_weights_path (str): Path to the classifier model weights (.pth).
        detector_weights_path (str): Path to the YOLOv8 detector weights (.pt).

    Returns:
        bool: True if models loaded successfully, False otherwise.
    """
    global classifier_model, detector_model, classifier_cam_generator
    global _loaded_classifier_name, _loaded_detector_path

    # Prevent reloading if already loaded with the same paths
    if classifier_model is not None and detector_model is not None and \
       _loaded_classifier_name == classifier_name and _loaded_detector_path == detector_weights_path:
       print("Pipeline models already loaded with the specified paths.")
       return True

    print("\n--- Loading Pipeline Models ---")

    # Load Classifier
    classifier_model = _load_saved_classifier(classifier_name, classifier_weights_path, CLASSIFIER_NUM_CLASSES, DEVICE)
    _loaded_classifier_name = classifier_name

    # Load Detector
    print(f"Loading detector from {detector_weights_path}...")
    if not os.path.exists(detector_weights_path):
        print(f"Error: Detector model weights not found at {detector_weights_path}")
        detector_model = None
        _loaded_detector_path = None
    else:
        try:
            detector_model = YOLO(detector_weights_path)
            print("Detector loaded successfully.")
            _loaded_detector_path = detector_weights_path
        except Exception as e:
            print(f"Error loading detector model: {e}")
            detector_model = None
            _loaded_detector_path = None

    # Initialize CAM generator if classifier loaded
    classifier_cam_generator = None # Reset in case of partial failure
    if classifier_model is not None:
       classifier_target_layer = _get_classifier_target_layer(classifier_name, classifier_model)
       if classifier_target_layer is not None:
           try:
              classifier_cam_generator = GradCAM(model=classifier_model, target_layers=[classifier_target_layer])
              print("GradCAM generator initialized.")
           except Exception as e:
              print(f"Error initializing GradCAM: {e}")
              classifier_cam_generator = None
       else:
            print("Skipping GradCAM initialization due to missing target layer.")
    else:
         print("Skipping GradCAM initialization as classifier model failed to load.")


    if classifier_model is not None and detector_model is not None and classifier_cam_generator is not None:
        print("--- Pipeline Models Loaded Successfully ---")
        return True
    else:
        print("--- Pipeline Model Loading Failed ---")
        return False


# --- Main Grad-CAM Generation Function (Call After Loading Models) ---
def generate_gradcam(
    image_np_rgb,
    detector_conf_threshold=DEFAULT_DETECTOR_CONF_THRESHOLD,
    detector_iou_threshold=DEFAULT_DETECTOR_IOU_THRESHOLD,
    apply_heatmap_filter=DEFAULT_APPLY_HEATMAP_FILTER,
    heatmap_threshold_ratio=DEFAULT_HEATMAP_THRESHOLD_RATIO
):
    """
    Processes an image (NumPy array) through the detection and classification pipeline
    and returns the Grad-CAM visualization, detected box, and classification results.

    Args:
        image_np_rgb (numpy array): The input thermal image as a NumPy array (H, W, C, RGB).
        detector_conf_threshold (float): Confidence threshold for object detection.
        detector_iou_threshold (float): IoU threshold for object detection NMS.
        apply_heatmap_filter (bool): Whether to threshold the CAM visualization.
        heatmap_threshold_ratio (float): Ratio for heatmap filtering if applied.

    Returns:
        tuple: (cam_overlaid_image, detected_box_pixel, detection_confidence_score, classifier_prediction_name, classifier_confidence_score)
               where cam_overlaid_image is a NumPy array (uint8 RGB) overlaid on the masked input,
               detected_box_pixel is a list [x_min, y_min, x_max, y_max] or None,
               detection_confidence_score (float),
               classifier_prediction_name (str),
               classifier_confidence_score (float).
               Returns (None, None, 0.0, "Models Not Loaded", 0.0) if models not loaded.
               Returns (None, None, 0.0, "Invalid Input", 0.0) if input is invalid.
               Returns (None, detected_box_pixel, detection_confidence_score, "No Box Detected", 0.0) if no box detected.
               Returns (None, detected_box_pixel, detection_confidence_score, "Processing Error", 0.0) if other errors occur.
    """
    # Check if models are loaded
    if classifier_model is None or detector_model is None or classifier_cam_generator is None:
        print("Error: Pipeline models are not loaded. Call load_pipeline_models() first.")
        return None, None, 0.0, "Models Not Loaded", 0.0

    # Input validation
    if not isinstance(image_np_rgb, np.ndarray) or image_np_rgb.ndim != 3 or image_np_rgb.shape[2] != 3:
        print("Error: Input image must be a 3-channel NumPy array (H, W, C, RGB).")
        return None, None, 0.0, "Invalid Input", 0.0

    img_height, img_width, _ = image_np_rgb.shape

    # --- Step 1: Run Object Detection (YOLOv8) ---
    detected_box_pixel = None
    detection_confidence_score = 0.0 # Initialize detection confidence

    try:
        # Pass the original image (NumPy array) to the detector
        detection_results = detector_model(image_np_rgb, conf=detector_conf_threshold, iou=detector_iou_threshold, verbose=False)

        if detection_results and detection_results[0].boxes:
            best_conf = 0
            best_box_idx = -1
            for i in range(len(detection_results[0].boxes.cls)):
                if int(detection_results[0].boxes.cls[i]) == 0: # Assuming 'breast' is class 0
                     conf = float(detection_results[0].boxes.conf[i])
                     if conf > best_conf:
                         best_conf = conf
                         best_box_idx = i

            if best_box_idx != -1:
                 box_coords = detection_results[0].boxes.xyxy[best_box_idx].cpu().numpy().astype(int).tolist()
                 detected_box_pixel = box_coords
                 detection_confidence_score = best_conf # Assign detection confidence here

        if detected_box_pixel is None:
            print("  Detector did NOT find a breast bounding box.")
            # Return with detection confidence score captured (even if 0)
            return None, detected_box_pixel, detection_confidence_score, "No Box Detected", 0.0

    except Exception as e:
        print(f"Error during object detection: {e}")
        # Return with detection confidence score captured (even if 0)
        return None, detected_box_pixel, detection_confidence_score, "Detection Error", 0.0


    # --- Step 3: Create Masked Image using Detected Bbox ---
    try:
        detected_box_list = pixel_to_corners_list(detected_box_pixel, class_id=None)
        masked_img_np_rgb = apply_masking(image_np_rgb, [detected_box_list], mask_type=MASK_TYPE)
    except Exception as e:
        print(f"Error during masking: {e}")
        return None, detected_box_pixel, detection_confidence_score, "Masking Error", 0.0


    # --- Step 4: Prepare Masked Image for Classification Model ---
    try:
         # The classifier model expects input transformed like the training data
         masked_img_tensor = classifier_transforms(masked_img_np_rgb).unsqueeze(0).to(DEVICE)
         # Ensure tensor requires grad for CAM generation
         masked_img_tensor.requires_grad_(True)
    except Exception as e:
        print(f"Error applying transforms to masked image: {e}")
        return None, detected_box_pixel, detection_confidence_score, "Transform Error", 0.0 # Return detection confidence


    # --- Step 5: Run Classification Model on Masked Image ---
    try:
        # No need for torch.no_grad() here as tensor requires grad for CAM
        classifier_outputs = classifier_model(masked_img_tensor)
        classifier_probs = torch.softmax(classifier_outputs, dim=1)[0]
        classifier_confidence_score = classifier_probs.max(0).values.item() # Get max value confidence
        classifier_predicted_class_idx = classifier_probs.argmax(0).item() # Get index
        classifier_predicted_class_name = CLASSIFIER_CLASS_NAMES.get(classifier_predicted_class_idx, "Unknown")

    except Exception as e:
        print(f"Error during classification inference: {e}")
        return None, detected_box_pixel, detection_confidence_score, "Classification Error", 0.0 # Return detection confidence


    # --- Step 6: Generate Grad-CAM for Classification Model on Masked Image ---
    try:
        classifier_targets = [ClassifierOutputTarget(classifier_predicted_class_idx)]

        # Generate heatmap for the masked image input
        grayscale_cam_masked = classifier_cam_generator(input_tensor=masked_img_tensor, targets=classifier_targets)[0, :]

        # Resize heatmap to original image dimensions for overlay visualization
        resized_heatmap_masked = cv2.resize(grayscale_cam_masked, (img_width, img_height))

    except Exception as e:
        print(f"Error during Grad-CAM generation: {e}")
        # Return with prediction results captured
        return None, detected_box_pixel, detection_confidence_score, classifier_predicted_class_name, classifier_confidence_score # Return detection confidence


    # --- Step 7: Filter Heatmap (Optional Visualization Step) ---
    if apply_heatmap_filter:
         filtered_heatmap_masked = _filter_heatmap_by_threshold(resized_heatmap_masked, threshold_ratio=heatmap_threshold_ratio)
         heatmap_to_overlay = filtered_heatmap_masked if filtered_heatmap_masked is not None else resized_heatmap_masked
    else:
         heatmap_to_overlay = resized_heatmap_masked


    # --- Step 8: Overlay Filtered CAM on Masked Image for Visualization ---
    try:
        masked_img_float = np.float32(masked_img_np_rgb) / 255
        cam_image_masked_overlay = show_cam_on_image(masked_img_float, heatmap_to_overlay, use_rgb=True)
    except Exception as e:
        print(f"Error during CAM overlay: {e}")
        # Return with prediction results captured
        return None, detected_box_pixel, detection_confidence_score, classifier_predicted_class_name, classifier_confidence_score # Return detection confidence


    # Return the visualization result and prediction info
    # Ensure all 5 values are returned
    return cam_image_masked_overlay, detected_box_pixel, detection_confidence_score, classifier_predicted_class_name, classifier_confidence_score


# --- Example Usage (for testing the module directly) ---
if __name__ == "__main__":
    # This block runs if you execute utils/gradcam.py directly
    # It's useful for testing the functions within this module
    # It's separate from how app.py will use it

    print("--- Testing utils/gradcam.py directly ---")

    # --- Step 1: Load Models ---
    # You might need to adjust these paths for your local setup if running this directly
    # If running this block, make sure you have models/efficientnet_b0_best_auc.pth and models/yolov8_breast_detector_yolov8_best.pt
    if not load_pipeline_models(
        classifier_weights_path='models/efficientnet_b0_best_auc.pth', # Example local path
        detector_weights_path='models/yolov8_breast_detector_yolov8_best.pt' # Example local path
        ):
        print("Failed to load models for direct testing. Exiting.")
        exit()

    # --- Step 2: Load a test image ---
    # Replace with a path to a test image on your system
    test_image_path = r'path/to/your/test_image.jpg'
    print(f"\nProcessing test image: {test_image_path}")

    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
    else:
        img_np = cv2.imread(test_image_path)
        if img_np is None:
            print(f"Error loading test image with OpenCV: {test_image_path}")
        else:
            # Convert to RGB if grayscale
            if len(img_np.shape) == 2: img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else: img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Ensure RGB

            # --- Step 3: Run the pipeline ---
            cam_overlay, detected_box, detection_conf, prediction_name, confidence_score = generate_gradcam(img_np_rgb)

            # --- Step 4: Display Results ---
            if cam_overlay is not None:
                 print(f"\nPipeline Results:")
                 print(f"  Predicted Class: {prediction_name} ({confidence_score:.4f})")
                 if detected_box:
                      print(f"  Detected Box (pixel): {detected_box}")
                      print(f"  Detection Confidence: {detection_conf:.4f}")
                 else:
                     print("  No box detected.")

                 # Display the CAM visualization image
                 plt.figure(figsize=(8, 6))
                 plt.imshow(cam_overlay)
                 plt.title(f"Grad-CAM for {os.path.basename(test_image_path)}\nPred: {prediction_name} ({confidence_score:.4f})")
                 plt.axis('off')
                 plt.show()
            else:
                 print("\nPipeline failed.")
                 print(f"Reason: {prediction_name}") # prediction_name holds the error message


    print("\n--- End of utils/gradcam.py test ---")
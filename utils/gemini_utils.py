# utils/gemini_utils.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
import cv2
import numpy as np
import base64
import json

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    raise ValueError("GEMINI_API_KEY not set in .env file")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured from utils.gemini_utils.")
except Exception as e:
    print(f"Error configuring Gemini API in utils.gemini_utils: {e}")
    raise e

# Use the model name you found works with image input (gemini-pro-vision for multimodal)
MULTIMODAL_MODEL_NAME = "gemini-2.5-flash-preview-04-17-thinking"


def get_gemini_model(model_name):
    """Helper to get a GenerativeModel instance."""
    try:
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        print(f"Error getting Gemini model {model_name}: {e}")
        return None


# --- Function to Generate Text Report from Model Results (Corrected Prompt Variable) ---
def generate_breast_cancer_report(model_results_text, patient_weight=None, patient_age=None, analysis_datetime_str=None, patient_name=None):
    """
    Generates a textual report for breast cancer detection based on model results and patient details.

    Args:
        model_results_text (str): A formatted string summarizing AI model predictions (e.g., healthy/sick, benign/malignant).
        patient_name (str, optional): Patient's name. Defaults to None.
        patient_age (str, optional): Patient's age. Defaults to None.
        analysis_datetime_str (str, optional): Date/time of the analysis. Defaults to None.

    Returns:
        dict: A dictionary containing the comprehensive and synthetic oncologist-style report.
    """
    model = get_gemini_model(MULTIMODAL_MODEL_NAME)
    if model is None:
        return {
            'comprehensive': 'Error: Gemini model not available for report generation.',
            'synthetic': 'Error: Gemini model not available for report generation.'
        }

    # Prepare patient details
    patient_info = ""
    if patient_name and patient_name.strip() and patient_name != 'N/A':
        patient_info += f"Patient Name: {patient_name.strip()}\n"
    if patient_age and patient_age.strip() and patient_age != 'N/A':
        patient_info += f"Patient Age: {patient_age.strip()}\n"
    if patient_weight and patient_weight.strip() and patient_weight != 'N/A':
        patient_info += f"Patient Weight: {patient_weight.strip()} kg\n"
    if analysis_datetime_str and analysis_datetime_str.strip():
        patient_info += f"Analysis Date/Time: {analysis_datetime_str.strip()}\n"

    if patient_info:
        patient_info = "Patient Information:\n" + patient_info + "\n"

    # Comprehensive oncologist-style report prompt
    prompt_comprehensive = f"""
    You are an AI assistant specialized in oncology. Generate a detailed breast cancer screening report for a patient based on AI model predictions from breast thermography or imaging.
    
    Begin with the patient’s information and the analysis date/time.
    Analyze the predictions provided (e.g., healthy/sick classification, benign/malignant classification, confidence scores).
    Explain what the findings suggest in terms of breast cancer risk, and if cancer is likely, comment on possible stage and clinical implications.
    Include a professional tone suitable for discussion with oncologists or specialists in a medical report.
    Mention the implications for further diagnostic steps (biopsy, MRI, etc.) if any abnormalities are suspected.

    {patient_info}

    AI Model Results:
    {model_results_text}

    Comprehensive Report:
    """

    # Synthetic quick summary for oncologist
    prompt_synthetic = f"""
    Generate a brief diagnostic impression of breast cancer screening based on the following AI model predictions and patient info.
    This summary should mimic what an oncologist might note after reviewing automated AI findings.
    State the likely classification (e.g., healthy, suspicious, benign lesion, malignant mass) and mention next steps if required.

    Patient Name: {patient_name.strip() if patient_name and patient_name.strip() and patient_name != 'N/A' else 'N/A'}
    Patient Age: {patient_age.strip() if patient_age and patient_age.strip() and patient_age != 'N/A' else 'N/A'}
    Analysis Time: {analysis_datetime_str.strip() if analysis_datetime_str and analysis_datetime_str.strip() else 'N/A'}

    Model Results:
    {model_results_text}

    Synthetic Impression:
    """

    try:
        print("Calling Gemini API for comprehensive report...")
        response_comprehensive = model.generate_content([prompt_comprehensive])
        comprehensive_report = response_comprehensive.text
        print("Comprehensive report generated.")
    except Exception as e:
        print(f"Error generating comprehensive report with Gemini: {e}")
        comprehensive_report = f"Error generating comprehensive report: {e}"

    try:
        print("Calling Gemini API for synthetic report...")
        response_synthetic = model.generate_content([prompt_synthetic])
        synthetic_report = response_synthetic.text
        print("Synthetic report generated.")
    except Exception as e:
        print(f"Error generating synthetic report with Gemini: {e}")
        synthetic_report = f"Error generating synthetic report: {e}"

    return {
        'comprehensive': comprehensive_report,
        'synthetic': synthetic_report
    }



# --- is_fundus_image_with_gemini function (Keep as is) ---
def is_thermal_image_with_gemini(image_path):
    # ... (Keep the existing code for is_fundus_image_with_gemini) ...
    """
    Checks if the uploaded image is a fundus image using Gemini API.
    ... (rest of the function) ...
    """
    model = get_gemini_model(MULTIMODAL_MODEL_NAME)
    if model is None:
        return False, "Gemini model not available for validation.", "Gemini model not available."

    try:
        img_np = cv2.imread(image_path)
        if img_np is None:
            print(f"Validation Error: Could not read image file {image_path} for Gemini validation.")
            return False, "Could not read image file.", "Failed to read uploaded image."

        is_success, buffer = cv2.imencode(".jpg", img_np)
        if not is_success:
             print(f"Validation Error: Could not encode image {image_path} to bytes for Gemini validation.")
             return False, "Could not encode image.", "Failed to process image for validation."

        image_part = {
            'mime_type': 'image/jpeg',
            'data': buffer.tobytes()
        }

    except Exception as e:
        print(f"Validation Error: Error preparing image {image_path} for Gemini API: {e}")
        return False, f"Error preparing image: {e}", "Internal error during image validation."

    prompt_parts = [
        image_part,
        "Is this a thermal image of an breast ? Answer only 'Yes' or 'No'."
    ]

    try:
        print(f"Calling Gemini API to validate image type: {os.path.basename(image_path)}...")
        response = model.generate_content(prompt_parts)
        gemini_response_text = response.text.strip().lower()
        print(f"Gemini validation response: '{gemini_response_text}'")

        is_fundus = gemini_response_text == 'yes'

        return is_fundus, response.text.strip(), None

    except Exception as e:
        print(f"Validation Error: Error calling Gemini API for validation: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback') and hasattr(e.response.prompt_feedback, 'block_reason'):
             block_reason = e.response.prompt_feedback.block_reason
             print(f"Validation Error: Prompt was blocked for reason: {block_reason}")
             return False, f"Validation blocked: {block_reason}", f"Image validation failed (blocked): {block_reason}"
        return False, f"API call error: {e}", "Internal error during image validation API call."
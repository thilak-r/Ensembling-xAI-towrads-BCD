import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        print(f"GradCAM initialized. Target layer: {type(self.target_layer)}")

        self.target_layer.register_forward_hook(self._save_activations)
        # Use register_full_backward_hook for newer PyTorch versions if register_backward_hook gives issues
        # For nn.Sequential, register_full_backward_hook is generally more robust
        self.target_layer.register_full_backward_hook(self._save_gradients) 
        print("Forward and backward hooks registered on target layer.")# Use full_backward_hook for newer PyTorch

    def _save_activations(self, module, input, output):
        self.activations = output
        print(f"Forward hook: Activations captured! Shape: {output.shape}")

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output is a tuple. We usually need the first element.
        self.gradients = grad_output[0] 
        print(f"Backward hook: Gradients captured! Shape: {grad_output[0].shape}")


    # In utils/gradcam.py
# class GradCAM:
# ...
    def generate_heatmap(self, model_output_logits): # model_output_logits is (batch_size, 1)
        if self.activations is None:
            print("Error: Activations not captured. Forward pass through target layer likely didn't occur after hook registration or hook failed.")
            # You can even print self.target_layer to see what it is
            return None
        
        # --- Backward Pass to get Gradients ---
        self.model.zero_grad() # Zero out any old gradients in the model

        # We need a scalar score to backpropagate from.
        # Assuming binary classification and model_output_logits is [N, 1]
        # We'll use the raw logit output for the class of interest.
        # For Grad-CAM, we typically want to see why the model made its prediction.
        # If model_output_logits > 0 (predicts positive), we use that score.
        # If model_output_logits < 0 (predicts negative), we might still use the positive class logit
        # or the negative one depending on what we want to visualize.
        # For simplicity, let's assume we're visualizing for the positive class prediction.
        
        score_for_backward = model_output_logits[:, 0] # Target the single output logit

        # If batch size is > 1, sum scores or pick one. For display, usually batch_size is 1.
        if score_for_backward.nelement() > 1 : # If more than one element (e.g. batch > 1)
             score_for_backward_scalar = score_for_backward.mean() # Or score_for_backward[0]
        else:
             score_for_backward_scalar = score_for_backward 

        # The backward call happens here. This should trigger the _save_gradients hook.
        score_for_backward_scalar.backward(retain_graph=True) # retain_graph can be helpful

        if self.gradients is None:
            print("Error: Gradients not captured after backward pass. Hook for gradients might have failed or backward pass didn't reach target layer.")
            return None

        # --- Process Gradients and Activations ---
        # Assuming self.gradients and self.activations are for a batch_size of 1 for visualization
        # self.gradients from hook: (B, C, H, W), typically B=1 here
        # self.activations from hook: (B, C, H, W), typically B=1 here

        grads_val = self.gradients[0]      # Shape: (C, H, W)
        activations_val = self.activations.detach()[0] # Shape: (C, H, W)

        pooled_gradients = torch.mean(grads_val, dim=[1, 2]) # Shape: (C,)

        # Weight activations by pooled gradients
        for i in range(activations_val.shape[0]): # Loop through channels (C)
            activations_val[i, :, :] *= pooled_gradients[i] 

        heatmap = torch.mean(activations_val, dim=0) # Average along channel. Shape: (H, W)
        heatmap = F.relu(heatmap) # Apply ReLU
        
        # Normalize heatmap
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        else: # Handle case of all-zero heatmap (e.g., if ReLU zeroes everything)
            print("Warning: Grad-CAM heatmap is all zeros after ReLU/normalization.")
            # Return a zero map of the correct spatial dimensions of the activations
            return np.zeros((self.activations.shape[2], self.activations.shape[3])) 
            
        return heatmap.cpu().numpy()


def overlay_gradcam(original_img_np, heatmap_np, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays the heatmap on the original image.
    original_img_np: NumPy array of the original image (H, W, C), range 0-255, uint8.
    heatmap_np: NumPy array of the heatmap (H, W), range 0-1.
    """
    if original_img_np.shape[:2] != heatmap_np.shape:
        heatmap_np = cv2.resize(heatmap_np, (original_img_np.shape[1], original_img_np.shape[0]))

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_np), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # Ensure RGB for display

    superimposed_img = cv2.addWeighted(original_img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img
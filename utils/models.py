import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --- Stage 1: Unimodal Models ---

# IMPORTANT: This CNN_Model definition MUST MATCH the one used for training
# best_kappa_model_cnn.pth
class CNN_Model(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1); self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1); self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1); self.bn3 = nn.BatchNorm2d(64)
        flat_size = self._get_conv_output_size((160, 120))
        self.fc1 = nn.Linear(flat_size, 512); self.fc_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes); self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x))); x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc_bn(self.fc1(x)))
        x_out = self.dropout(features); x_out = torch.sigmoid(self.fc2(x_out))
        return x_out
    def _get_conv_output_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, 3, *input_size)
            x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool2d(x, 2)
            x = F.relu(self.bn3(self.conv3(x))); x = F.max_pool2d(x, 2)
            return x.view(x.size(0), -1).size(1)
    def get_features(self, x): # For t-SNE
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x))); x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1); features = F.relu(self.fc_bn(self.fc1(x)))
        return features


def get_torchvision_model(model_name, num_classes=1, pretrained=False):
    """
    Loads a pretrained torchvision model and replaces its classifier.
    For deployment, pretrained should ideally be False if you are loading your own fine-tuned weights,
    unless your saved state_dict *only* contains the final layer.
    However, to match training, if you loaded weights=DEFAULT and then only trained the FC,
    you might need weights=DEFAULT here too, then load your fine-tuned state_dict over it.
    For simplicity, assuming state_dict contains all necessary weights.
    """
    model = None
    if model_name == "resnet18":
        model = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT) # Or specify weights version
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None if not pretrained else models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None if not pretrained else models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not recognized for torchvision.")
    return model


# In utils/models.py

# ... (Your CNN_Model and get_torchvision_model function should be above this) ...

class MultimodalBreastCancerCNN(nn.Module):
    def __init__(self, num_clinical_features, num_classes=1, dropout_rate=0.4):
        super(MultimodalBreastCancerCNN, self).__init__()
        
        # Image Branch (using ResNet18) - AS PER YOUR ORIGINAL TRAINING
        self.image_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs_img = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity() # Remove original FC layer

        # NO self.grad_cam_target_layer definition here in __init__

        # Clinical Data Branch (remains the same)
        self.clinical_mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusion and Classifier Head (remains the same)
        self.fusion_dropout = nn.Dropout(dropout_rate + 0.1) 
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs_img + 32, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes) 
        )

    def forward(self, image_input, clinical_input):
        # --- Image Branch - Modified for Grad-CAM ---
        # Pass through ResNet backbone layers individually using 'self.image_model'
        x = self.image_model.conv1(image_input)
        x = self.image_model.bn1(x)
        x = self.image_model.relu(x)
        x = self.image_model.maxpool(x)

        x = self.image_model.layer1(x)
        x = self.image_model.layer2(x)
        x = self.image_model.layer3(x)
        image_conv_features = self.image_model.layer4(x) # Output of self.image_model.layer4
        
        # Continue to get final image features for classification
        x_for_classification = self.image_model.avgpool(image_conv_features)
        image_features_flat = torch.flatten(x_for_classification, 1)
        
        # --- Clinical Branch ---
        clinical_features = self.clinical_mlp(clinical_input)
        
        # --- Fusion ---
        fused_features = torch.cat((image_features_flat, clinical_features), dim=1)
        fused_features = self.fusion_dropout(fused_features)
        
        # --- Classifier ---
        output_logits = self.classifier(fused_features)
        
        # Return both logits AND the conv features for Grad-CAM
        return output_logits, image_conv_features
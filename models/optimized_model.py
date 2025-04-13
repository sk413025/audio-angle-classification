import torch
import torch.nn as nn
import torchvision.models as models

class OptimizedResNetClassifier(nn.Module):
    """
    Optimized ResNet classifier with improved efficiency and accuracy.
    """
    def __init__(self, num_classes=360, pretrained=True, backbone='resnet18'):
        super(OptimizedResNetClassifier, self).__init__()
        
        # Select backbone architecture
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Initialize the weights
        self._initialize_weights()
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# Specialized model for angle classification with better precision
class AngleClassifier(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super(AngleClassifier, self).__init__()
        
        # Base feature extractor
        self.feature_extractor = OptimizedResNetClassifier(
            num_classes=360,
            pretrained=pretrained,
            backbone=backbone
        )
        
        # Replace the final layer with specialized angle prediction
        feature_dim = 256  # From the OptimizedResNetClassifier
        self.feature_extractor.classifier = nn.Sequential(
            *list(self.feature_extractor.classifier.children())[:-1]
        )
        
        # Angle-specific prediction heads
        self.coarse_angle = nn.Linear(feature_dim, 36)  # 10-degree precision
        self.fine_angle = nn.Linear(feature_dim, 10)    # 1-degree precision
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        coarse = self.coarse_angle(features)
        fine = self.fine_angle(features)
        
        # Convert to angle prediction (0-359 degrees)
        pred_coarse = torch.argmax(coarse, dim=1) * 10
        pred_fine = torch.argmax(fine, dim=1)
        
        angle_pred = pred_coarse + pred_fine
        
        return {
            'coarse': coarse,
            'fine': fine,
            'angle': angle_pred
        } 
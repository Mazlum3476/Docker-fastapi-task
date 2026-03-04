import torch.nn as nn
from torchvision import models

class ObjectDetector(nn.Module):
    def __init__(self, num_classes=2): 
        super(ObjectDetector, self).__init__()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.regressor = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4), nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        bboxes = self.regressor(x)
        class_logits = self.classifier(x)
        return (bboxes, class_logits)
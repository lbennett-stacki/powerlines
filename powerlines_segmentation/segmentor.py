import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
import cv2
import matplotlib.pyplot as plt


class Segmentor:
    def __init__(self):
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        # 2 classes: background, person
        self.model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)

    def segment_person(self, image_path, model):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (512, 512))
        input_tensor = (
            torch.tensor(resized_image.transpose(2, 0, 1), dtype=torch.float32)
            .unsqueeze(0)
            .cuda()
            / 255.0
        )

        with torch.no_grad():
            output = model(input_tensor)["out"]
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        return resized_image, mask

    def segment(self, image_path: str):
        self.model.load_state_dict(torch.load("deeplab_coco_person.pth"))
        self.model.eval()

        image, mask = self.segment_person(image_path, self.model)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.title("Person Mask")
        plt.imshow(mask, cmap="jet")
        plt.show()

import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
import cv2
from tqdm import tqdm
from powerlines_segmentation.dataset import COCOSegmentationDataset
from torch.amp import GradScaler, autocast


class Trainer:
    def __init__(self):
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        # 2 classes: background, person
        self.model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
        self.model.cuda()
        self.scaler = GradScaler("cuda")

    def train(self, epochs=10):
        coco_root = "data/coco/train2017"
        coco_annotations = "data/coco/annotations/instances_train2017.json"
        dataset = COCOSegmentationDataset(
            coco_root, coco_annotations, transform=self.transform
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0
            for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, masks = images.cuda(), masks.cuda()
                optimizer.zero_grad()

                with autocast("cuda"):
                    outputs = self.model(images)["out"]
                    loss = criterion(outputs, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

        torch.save(self.model.state_dict(), "deeplab_coco_person.pth")

    def transform(self, image, mask):
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        return image, mask

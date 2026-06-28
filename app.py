import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(PATHOLOGIES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
THRESHOLD = 0.45

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_model(model_path: str = "best_densenet121.pth") -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = build_model("best_densenet121.pth")


class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.features.denseblock4
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(tensor)
        score = logits[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam = cv2.resize(cam, (224, 224))
        return cam.astype(np.float32)


gradcam = GradCAM(model)


def overlay_heatmap(original_pil: Image.Image, cam: np.ndarray) -> Image.Image:
    base = np.array(original_pil.convert("RGB").resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * base + 0.45 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)


def predict(image: Image.Image):
    img_rgb = image.convert("RGB")
    tensor = val_transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().tolist()

    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    top_idx, top_prob = ranked[0]

    if top_prob < THRESHOLD:
        return {"No Finding": 1.0}, None

    label_dict = {
        p.replace("_", " "): round(probs[i], 4)
        for i, p in enumerate(PATHOLOGIES)
    }

    tensor_grad = val_transform(img_rgb).unsqueeze(0)
    tensor_grad.requires_grad_(True)
    cam = gradcam.generate(tensor_grad, top_idx)
    overlay = overlay_heatmap(img_rgb, cam)

    return label_dict, overlay


with gr.Blocks(title="Chest X-Ray Classifier") as demo:
    gr.Markdown(
        """
        # 🫁 Chest X-Ray Disease Classifier
        **DenseNet-121** · NIH ChestX-ray14 · 14 thoracic diseases

        If the highest disease confidence is below 45%, the app returns **No Finding**
        and skips Grad-CAM.

        """
    )

    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Chest X-Ray")

    with gr.Row():
        label_output = gr.Label(num_top_classes=14, label="Disease Probabilities")
        cam_output = gr.Image(type="pil", label="Grad-CAM Heatmap")

    run_btn = gr.Button("Analyze", variant="primary")
    run_btn.click(fn=predict, inputs=img_input, outputs=[label_output, cam_output])
    img_input.change(fn=predict, inputs=img_input, outputs=[label_output, cam_output])

if __name__ == "__main__":
    demo.launch()

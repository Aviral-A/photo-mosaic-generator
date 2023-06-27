# embeddings_utils.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class EmbeddingsModel:
    def __init__(self):
        self.model = self.load_model()
        self.device = self.check_mps_availability()
        self.model = self.set_device(self.model, self.device)

    def load_model(self):
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        return model

    def check_mps_availability(self):
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                      "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                      "and/or you do not have an MPS-enabled device on this machine.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("mps")
        return device

    def set_device(self, model, device):
        model = model.to(device)
        model.eval()
        return model

    def compute_embeddings(self, image_path) -> torch.Tensor:
        image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embeddings = self.model(image)

        return embeddings.squeeze().cpu().numpy()
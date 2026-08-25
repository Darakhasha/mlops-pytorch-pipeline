from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model


app = FastAPI(
    title="MLOps PyTorch Classifier",
    version="1.0.0",
)


# CIFAR-10 class names
CLASS_NAMES = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
]


def load_config():

    config_path = Path("/app/configs/training_config.yaml")

    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model():

    config = load_config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    )

    checkpoint_path = (
        Path(config["output"]["checkpoint_dir"])
        / config["output"]["model_name"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, device


# Load model when application starts
try:
    model, device = load_model()
    MODEL_LOADED = True
    MODEL_ERROR = None

except Exception as exc:
    model = None
    device = torch.device("cpu")
    MODEL_LOADED = False
    MODEL_ERROR = str(exc)

image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.2860],
        std=[0.3530],
    ),
])


@app.get("/health")
def health():

    if MODEL_LOADED:
        return {
            "status": "healthy",
            "model_loaded": True,
        }

    raise HTTPException(
        status_code=503,
        detail={
            "status": "unhealthy",
            "model_loaded": False,
            "error": MODEL_ERROR,
        },
    )


@app.post("/predict")
async def predict(
    image: UploadFile = File(...)
):

    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:

        image_bytes = await image.read()

        pil_image = Image.open(
            __import__("io").BytesIO(image_bytes)
        ).convert("RGB")

        input_tensor = image_transform(
            pil_image
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[0]

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

        return {
            "predicted_class": CLASS_NAMES[
                predicted_index
            ],
            "class_index": predicted_index,
            "probabilities": {
                CLASS_NAMES[i]: round(
                    float(probabilities[i]),
                    6,
                )
                for i in range(len(CLASS_NAMES))
            },
        }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        )
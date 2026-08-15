from src.utils.config import get_config, get_model_config


config = get_config()
model_config = get_model_config()

print("Project:", config["project"]["name"])
print("Image size:", config["data"]["image_size"])
print("Number of slices:", config["data"]["num_slices"])
print("Model:", model_config["model"]["architecture"])
print("Classes:", model_config["model"]["num_classes"])
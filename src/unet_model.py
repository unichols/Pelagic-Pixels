import segmentation_models_pytorch as smp

def get_unet():
    return smp.Unet(
        encoder_name="mobilenet_v2",  # Lightweight backbone
        encoder_weights="imagenet",  # Pre-trained on ImageNet
        in_channels=3,               # RGB images
        classes=1                    # Binary segmentation
    )

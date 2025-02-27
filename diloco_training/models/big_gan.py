import torchvision.models as models


def get_biggan(pretrained=True):
    """Return a BigGAN model"""
    model = models.biggan_b16(pretrained=pretrained)
    return model


# Example usage
if __name__ == "__main__":
    model = get_biggan()
    print(model)

import timm


def build_model(num_classes: int, pretrained: bool = True, drop_rate: float = 0.2):
    model = timm.create_model(
        "mobilenetv4_conv_small.e1200_r224_in1k",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model

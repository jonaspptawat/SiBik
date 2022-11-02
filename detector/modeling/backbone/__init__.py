from .shufflenetv2 import ShuffleNetV2

def get_backbone():
    # If you want to train, you will need to add "cfg"
    model = ShuffleNetV2()
    return model

from .head import Detector

__all__ = ["Detector", "build_detector"]

def build_detector(classes):
    detector = Detector(classes=classes)
    return detector

if __name__ == "__main__":
    print("Hello")

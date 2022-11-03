from .build import YOLODataset

def build_dataset(cfg, path, transform):
    dataset = YOLODataset(cfg, path, transform)
    return dataset

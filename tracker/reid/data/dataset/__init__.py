from .build import REIDdataset, REIDTest

def build_dataset(cfg, path, transform):
    dataset = REIDdataset(cfg, path, transform)
    return dataset

def build_testset(path, transform):
    dataset = REIDTest(path, transform)
    return dataset

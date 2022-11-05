from .dataset import build_dataset, build_testset
from .transform import build_transforms
from torch.utils.data import DataLoader


def build_dataloader(cfg, is_train=True, **kwargs):
    
    transformer = build_transforms(cfg, is_train)

    # Add more config here
    if is_train:
        path = cfg.DATA.TRAIN
        batch = cfg.DATA.SIZE_TRAIN
        drop_last = True
        dataset = build_dataset(cfg, path, transform=transformer)
    else:
        path = cfg.DATA.VAL
        batch = cfg.DATA.SIZE_VAL
        drop_last = False
        dataset = build_testset(path, transform=transformer)

    dataloader = DataLoader(dataset,
                            batch_size=batch,
                            drop_last=drop_last,
                            **kwargs)
    
    return dataloader

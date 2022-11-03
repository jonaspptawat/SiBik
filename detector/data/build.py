from .datasets import build_dataset
from .transforms import build_transforms
from .collate_batch import collate_fn
from torch.utils.data import DataLoader

def build_dataloader(cfg, is_train=True, **kwargs):

    transformer = build_transforms(cfg, is_train)

    # Add more config here
    if is_train:
        path = cfg.DATA.TRAIN
        batch = cfg.DATA.SIZE_TRAIN
        drop_last = True
    else:
        path = cfg.DATA.VAL
        batch = cfg.DATA.SIZE_VAL
        drop_last = False

    dataset = build_dataset(cfg, path, transformer)
    dataloader = DataLoader(dataset,
                            batch_size=batch,
                            collate_fn=collate_fn,
                            drop_last=drop_last,
                            **kwargs)

    return dataloader

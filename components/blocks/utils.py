import random


def random_crop(x, crop_size=128):
    if x.ndim == 4:
        _, _, H, W = x.shape
        h = random.randint(0, H - crop_size)
        w = random.randint(0, W - crop_size)
        x = x[:, :, h : h + crop_size, w : w + crop_size]
    elif x.ndim == 5:
        _, _, _, H, W = x.shape
        h = random.randint(0, H - crop_size)
        w = random.randint(0, W - crop_size)
        x = x[:, :, :, h : h + crop_size, w : w + crop_size]
    else:
        raise RuntimeError(
            f"Expected 4D (non-temporal) or 5D (temporal) input, but got input of size: {list(x.shape)}"
        )
    return x

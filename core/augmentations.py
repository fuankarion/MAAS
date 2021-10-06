import random

from PIL import Image
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import resized_crop


def video_temporal_crop(video_data, crop_ratio):
    # random flip
    if bool(random.getrandbits(1)):
        video_data = [s.transpose(Image.FLIP_LEFT_RIGHT) for s in video_data]

    # random crop
    mid = int(len(video_data) / 2)
    width, height = video_data[mid].size
    f = random.uniform(crop_ratio, 1)
    i, j, h, w = RandomCrop.get_params(video_data[mid], output_size=(int(height*f), int(width*f)))
    video_data = [s.crop(box=(j, i, j+w, i+h)) for s in video_data]
    return video_data


def video_corner_crop(video_data, crop_ratio):
    # random flip
    if bool(random.getrandbits(1)):
        video_data = [s.transpose(Image.FLIP_LEFT_RIGHT) for s in video_data]

    corner = random.randint(0, 4)  # Random Corner is the same for the tube
    for v in video_data:
        width, height = v.size
        new_w = int(width*crop_ratio)
        new_h = int(height*crop_ratio)

        if corner == 0:
            v = resized_crop(v, 0, 0, new_h, new_w, (new_h, new_w))
        elif corner == 1:
            v = resized_crop(v, height-new_h, 0, height, new_w, (new_h, new_w))
        elif corner == 2:
            v = resized_crop(v, 0, width-new_w, new_h, width, (new_h, new_w))
        elif corner == 3:
            v = resized_crop(v, height-new_h, width-new_w, height, width, (new_h, new_w))
        elif corner == 4:
            pass  # full image
        else:
            print('Invalid corner')

    return video_data

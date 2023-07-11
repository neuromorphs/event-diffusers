# We have to do this to ignore stupid SSL errors
import ssl
from pathlib import Path

import tonic
import torch
from datasets import Dataset, load_dataset, load_from_disk
from torchvision import transforms

from .config import TrainingConfig

config = TrainingConfig()

gesture_path = Path("data/gesture_dataset")

ssl._create_default_https_context = ssl._create_unverified_context

#################
# DAVIS Dataset #
#################

dataset = tonic.datasets.DAVISDATA(save_to="data", recording="shapes_6dof")
data, targets = dataset[0]
events, imu, images = data
frame_time = images["ts"][1] - images["ts"][0]

# You will need at least Tonic 1.3.2 for this to work!
tau = frame_time / 10
sufarce_transform1 = tonic.transforms.ToTimesurface(
    sensor_size=dataset.sensor_size, tau=tau, dt=frame_time
)
sufarce_transform2 = tonic.transforms.ToTimesurface(
    sensor_size=dataset.sensor_size, tau=10 * tau, dt=frame_time
)
sufarce_transform3 = tonic.transforms.ToTimesurface(
    sensor_size=dataset.sensor_size, tau=100 * tau, dt=frame_time
)


def data_transform(data):
    events, imu, images = data
    surfaces1 = sufarce_transform1(events)
    surfaces2 = sufarce_transform2(events)
    surfaces3 = sufarce_transform3(events)
    return surfaces1, surfaces2, surfaces3, imu, images


davis_dataset = tonic.datasets.DAVISDATA(
    save_to="data", recording="shapes_6dof", transform=data_transform
)

#####################
# Butterfly Dataset #
#####################

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

butterfly_dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


butterfly_dataset.set_transform(transform)

###################
# Gesture Dataset #
###################

trainset = tonic.datasets.DVSGesture("data", train=True)

frame_time = 200_000  # microseconds
slicer = tonic.slicers.SliceByTime(time_window=frame_time)
transform = tonic.transforms.ToTimesurface(
    sensor_size=trainset.sensor_size, dt=frame_time, tau=frame_time / 10
)
gesture_dataset = tonic.SlicedDataset(
    trainset,
    slicer=slicer,
    metadata_path=f"metadata/slicing/{frame_time}",
    transform=transform,
)

if gesture_path.exists():
    gesture_dataset = load_from_disk(gesture_path)
else:
    data_list = [
        {"data": torch.Tensor(d[0]), "label": torch.Tensor(d[1])}
        for d in gesture_dataset
    ]
    hf_dataset = Dataset.from_list(data_list)
    hf_dataset.save_to_disk(gesture_path)
    gesture_dataset = hf_dataset
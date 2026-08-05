import numpy as np
import os
from tqdm import tqdm
import torchvision

class ETSMnistData:
    def __init__(self, time_major, pad_size=256):
        self.threshold = 128
        self.pad_size = pad_size

        if not self.load_from_cache():
            self.create_dataset()

        self.train_elapsed /= self.pad_size
        self.test_elapsed /= self.pad_size

    def load_from_cache(self):
        if os.path.isfile("dataset/test_mask.npy"):
            self.train_events = np.load("dataset/train_events.npy")
            self.train_elapsed = np.load("dataset/train_elapsed.npy")
            self.train_mask = np.load("dataset/train_mask.npy")
            self.train_y = np.load("dataset/train_y.npy")

            self.test_events = np.load("dataset/test_events.npy")
            self.test_elapsed = np.load("dataset/test_elapsed.npy")
            self.test_mask = np.load("dataset/test_mask.npy")
            self.test_y = np.load("dataset/test_y.npy")

            if os.path.isfile("dataset/test_x.npy"):
                self.test_x = np.load("dataset/test_x.npy")
                self.train_x = np.load("dataset/train_x.npy")
            else:
                self.test_x = None
                self.train_x = None

            print("train_events.shape: ", str(self.train_events.shape))
            print("train_elapsed.shape: ", str(self.train_elapsed.shape))
            print("train_mask.shape: ", str(self.train_mask.shape))
            print("train_y.shape: ", str(self.train_y.shape))

            print("test_events.shape: ", str(self.test_events.shape))
            print("test_elapsed.shape: ", str(self.test_elapsed.shape))
            print("test_mask.shape: ", str(self.test_mask.shape))
            print("test_y.shape: ", str(self.test_y.shape))
            return True
        return False

    def transform_sample(self, x):
        x = x.flatten()

        events = np.zeros([self.pad_size], dtype=np.float32)
        elapsed = np.zeros([self.pad_size, 1], dtype=np.float32)
        mask = np.zeros([self.pad_size], dtype=np.bool)

        last_char = -1
        write_index = 0
        elapsed_counter = 0
        for i in range(x.shape[0]):
            elapsed_counter += 1
            char = int(x[i] > self.threshold)
            if last_char != char:
                events[write_index] = char
                elapsed[write_index] = elapsed_counter
                mask[write_index] = True
                write_index += 1
                if write_index >= self.pad_size:
                    # Enough 1s in this sample, abort
                    self._abort_counter += 1
                    break
                elapsed_counter = 0
            last_char = char
        self._all_lenghts.append(write_index)
        return events, elapsed, mask

    def transform_array(self, x):
        events_list = []
        elapsed_list = []
        mask_list = []

        for i in tqdm(range(x.shape[0])):
            events, elapsed, mask = self.transform_sample(x[i])
            events_list.append(events)
            elapsed_list.append(elapsed)
            mask_list.append(mask)

        return (
            np.stack(events_list, axis=0),
            np.stack(elapsed_list, axis=0),
            np.stack(mask_list, axis=0),
        )

    def create_dataset(self):
        mnist_train = torchvision.datasets.MNIST(
            root="./data", train=True, download=True
        )
        mnist_test = torchvision.datasets.MNIST(
            root="./data", train=False, download=True
        )

        train_x = mnist_train.data.numpy()
        train_y = mnist_train.targets.numpy().astype(np.uint8)
        test_x = mnist_test.data.numpy()
        test_y = mnist_test.targets.numpy().astype(np.uint8)

        self._all_lenghts = []
        self._abort_counter = 0

        train_x = train_x.reshape([-1, 28 * 28])
        test_x = test_x.reshape([-1, 28 * 28])

        self.train_y = train_y
        self.test_y = test_y

        print("Transforming training samples")
        self.train_events, self.train_elapsed, self.train_mask = self.transform_array(
            train_x
        )
        print("Transforming test samples")
        self.test_events, self.test_elapsed, self.test_mask = self.transform_array(
            test_x
        )

        print("Average time-series length: {:0.2f}".format(np.mean(self._all_lenghts)))
        print("Abort counter: ", str(self._abort_counter))
        os.makedirs("dataset", exist_ok=True)
        np.save("dataset/train_events.npy", self.train_events)
        np.save("dataset/train_elapsed.npy", self.train_elapsed)
        np.save("dataset/train_mask.npy", self.train_mask)
        np.save("dataset/train_y.npy", self.train_y)

        np.save("dataset/test_events.npy", self.test_events)
        np.save("dataset/test_elapsed.npy", self.test_elapsed)
        np.save("dataset/test_mask.npy", self.test_mask)
        np.save("dataset/test_y.npy", self.test_y)

        np.save("dataset/train_x.npy", train_x)
        np.save("dataset/test_x.npy", test_x)


import torch
from torch.utils.data import Dataset

class ETSMnistDataset(Dataset):
    def __init__(self, data_obj: ETSMnistData, mode: str = 'train'):
        self.data_obj = data_obj
        if mode == 'train':
            self.events = torch.from_numpy(data_obj.train_events).float()
            self.ts = torch.from_numpy(data_obj.train_elapsed).float()
            self.mask = torch.from_numpy(data_obj.train_mask).bool()
            self.labels = torch.from_numpy(data_obj.train_y).long()
        elif mode == 'test':
            self.events = torch.from_numpy(data_obj.test_events).float()
            self.ts = torch.from_numpy(data_obj.test_elapsed).float()
            self.mask = torch.from_numpy(data_obj.test_mask).bool()
            self.labels = torch.from_numpy(data_obj.test_y).long()
        else:
            raise ValueError("Mode must be 'train' or 'test'")

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            "events": self.events[idx], # [pad_size, 1]
            "ts": self.ts[idx],         # [pad_size, 1]
            "mask": self.mask[idx],     # [pad_size]
            "label": self.labels[idx]   # scalar
        }

    def get_raw_image(self, idx: int) -> np.ndarray:
        """Возвращает оригинальное изображение в формате numpy [28, 28]."""
        if idx < 0 or idx >= len(self.data_obj.test_y):  # Упрощенно для теста
            raise IndexError("Index out of range")

        img = self.data_obj.test_x[idx].reshape(28, 28)
        return img
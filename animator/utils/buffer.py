from random import randint, uniform
from warnings import warn

import torch
from torch.distributed import all_gather


class ImageBuffer():
    def __init__(self, worldsize: int, size: int = 50) -> None:
        if size < worldsize:
            raise ValueError('size should be bigger than worldsize')
        if size % worldsize != 0:
            warn(' '.join(['buffer size % worldsize != 0,',
                           '{} images generated during the last'.format(worldsize - size % worldsize),
                           'populating buffer step will be omitted']))

        self.size = size
        self.worldsize = worldsize
        self.storage = []
        self.buffer = None
        self.ind = -1
        self.device = None
        self.shape = None

    def add(self, img: torch.Tensor) -> None:
        self.buffer = img
        if self.shape is None:
            self.shape = self.buffer.shape
        if self.device is None:
            self.device = self.buffer.device

    def get(self,) -> torch.Tensor:
        """
        By 50% chance, the buffer will return last imput image/images.
        By 50% chance, the buffer will return image/images previously stored in the buffer
        """
        if self.buffer is None:
            raise RuntimeError('method "add" should be called before "get"')
        # Get random number from uniform distribution on the interval [0, 1)
        prob = uniform(0, 1)
        img = self.buffer.clone()
        if prob > 0.5 and len(self.storage) == self.size:
            self.ind = randint(0, len(self.storage) - 1)
            img = self.storage[self.ind]
            self.storage[self.ind] = self.buffer
        elif len(self.storage) < self.size:
            self.ind = -1
        return img

    def step(self,) -> None:
        """Generate images cross-buffer exchange."""
        image_list = [torch.empty(self.shape,
                                  device = self.device,
                                  dtype = self.buffer.dtype) for _ in range(self.worldsize)]
        index_list = [torch.empty(1,
                                  device=self.device,
                                  dtype = torch.int32) for _ in range(self.worldsize)]
        all_gather(image_list, self.buffer)
        all_gather(index_list, torch.tensor([self.ind], device = self.device, dtype = torch.int32))
        for ind, ten in zip(index_list, image_list):
            if ind.item() == -1 and len(self.storage) < self.size:
                self.storage.append(ten)
            elif ind.item() >= 0:
                # if several processes change the same index apply the last one
                self.storage[ind] = ten
        self.ind = -2
        self.buffer = None

    def state_dict(self,) -> dict[str, int | list[torch.tensor] | torch.device | torch.Size | None]:
        state = {}
        state['size'] = self.size
        state['ind'] = self.ind
        state['shape'] = self.shape
        state['storage'] = [ten.clone().to('cpu') for ten in self.storage]
        return state

    def load_state_dict(self, state:
                        dict[str, int | list[torch.tensor] | torch.device | torch.Size | None]) -> None:
        for key, val in state.items():
            if key != 'storage':
                self.__dict__[key] = val
            else:
                self.__dict__[key] = [ten.to(self.device) for ten in val]

import torch
from random import randint, uniform

class ImageBuffer():
    def __init__(self, size: int = 50) -> None:
        if size == 0:
            raise ValueError('size should be bigger than 0')

        self.size = size
        self.ind = -1
        self.storage = []
        self.buffer = None

    def add(self, img: torch.Tensor) -> None:
        if self.ind + 1 < self.size:
            self.ind += 1
        else:
            self.ind = 0
        self.buffer = img
    
    def get(self,) -> torch.Tensor:
        '''
        By 50% chance, the buffer will return last imput image/images.
        By 50% chance, the buffer will return image/images previously stored in the buffer
        '''
        if self.buffer is None:
            raise RuntimeError('method "add" should be called before "get"')
        prob = uniform(0, 1)
        img = self.buffer
        if prob > 0.5 and len(self.storage) > 0:                 
            ind = randint(0, len(self.storage) - 1)
            img = self.storage[ind]

        if self.ind == len(self.storage):
            self.storage.append(self.buffer)
        else:
            self.storage[self.ind] = self.buffer
        self.buffer = None
        return img  




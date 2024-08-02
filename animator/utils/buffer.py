import torch
from random import randint, uniform
from collections import deque

class ImageBuffer():
    def __init__(self, generator: torch.Generator, size: int = 50) -> None:
        if size == 0:
            raise ValueError('size should be bigger than 0')

        self.size = size
        self.storage = []
        self.buffer = None
        self.generator = generator
        self.queue = deque()

    def add(self, img: torch.Tensor) -> None:
        self.buffer = img
    
    def get(self,) -> torch.Tensor:
        '''
        By 50% chance, the buffer will return last imput image/images.
        By 50% chance, the buffer will return image/images previously stored in the buffer
        '''
        if self.buffer is None:
            raise RuntimeError('method "add" should be called before "get"')
        # Get random number from uniform distribution on the interval [0, 1)
        prob = torch.rand(1, generator = self.generator).item()
        img = self.buffer
        if prob > 0.5 and len(self.storage) == self.size:                 
            ind = torch.randint(0, len(self.storage), (1,), generator = self.generator).item()
            img = self.storage[ind]
            self.storage[ind] = self.buffer
            self.queue.append(ind)
        elif len(self.storage) < self.size:
            self.storage.append(self.buffer)
            self.queue.append(ind)
        self.buffer = None
        return img
     
    def step():
        "Unfinished: values change and initial index fill logic"

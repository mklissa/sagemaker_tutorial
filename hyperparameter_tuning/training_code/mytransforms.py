
from mxnet.gluon.nn import Sequential,HybridSequential
from mxnet.gluon.block import Block,HybridBlock 
import numbers
import collections
import numpy as np
import mxnet as mx
from mxnet import nd

class Compose(Sequential):
    """Sequentially composes multiple transforms.
    Parameters
    ----------
    transforms : list of transform Blocks.
        The list of transforms to be composed.
    Inputs:
        - **data**: input tensor with shape of the first transform Block requires.
    Outputs:
        - **out**: output tensor with shape of the last transform Block produces.
    Examples
    --------
    >>> transformer = transforms.Compose([transforms.Resize(300),
    ...                                   transforms.CenterCrop(256),
    ...                                   transforms.ToTensor()])
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 3x256x256 @cpu(0)>
    """
    def __init__(self, transforms):
        super(Compose, self).__init__()
        transforms.append(None)
        hybrid = []
        for i in transforms:
            if isinstance(i, HybridBlock):
                hybrid.append(i)
                continue
            elif len(hybrid) == 1:
                self.add(hybrid[0])
                hybrid = []
            elif len(hybrid) > 1:
                hblock = HybridSequential()
                for j in hybrid:
                    hblock.add(j)
                hblock.hybridize()
                self.add(hblock)
                hybrid = []

            if i is not None:
                self.add(i)
                

class RandomCrop(Block):


    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
     
        super(RandomCrop, self).__init__()

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        img= img.asnumpy()
        
        if self.padding is not None:
            img = pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            
#         import pdb;pdb.set_trace()
        
        img = mx.image.random_crop(nd.array(img), self.size)[0]
        return img
    

    
def pad(img, padding, fill=0, padding_mode='constant'):
    r"""Pad the given PIL Image on all sides with specified padding mode and fill value.
    Args:
        img (PIL Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        PIL Image: Padded image.
    """
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

#     if padding_mode == 'constant':
#         return ImageOps.expand(img, border=padding, fill=fill)
#     else:

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

#     img = np.asarray(img)
    # RGB image
    if len(img.shape) == 3:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
    # Grayscale image
    if len(img.shape) == 2:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

    return img   
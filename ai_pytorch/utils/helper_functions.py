from unnormalize import UnNormalize
import matplotlib.pyplot as plt
import numpy as np


class HelperFunctions(object):

    def __init__(self):
        pass

    def imshow(self, img, text=None):
        unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        unorm(img)
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(
                75, 8, text, style='italic', fontweight='bold',
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10}
            )
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def show_plot(self, iteration, loss):
        plt.plot(iteration, loss)
        plt.show()

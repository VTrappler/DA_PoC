from moviepy.editor import ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage
import tqdm
import matplotlib.pyplot as plt


def create_mp4(array, plot_func, fps=10):
    seq = []
    for ar in tqdm.tqdm(array):
        # fig, axs = plt.subplots()
        fig = plot_func(array)
        # axs.imshow(ar)
        npi = mplfig_to_npimage(fig)
        plt.close()
        seq.append(npi)
    animation = ImageSequenceClip(seq, fps=fps)
    return animation.ipython_display(loop=True, autoplay=True)

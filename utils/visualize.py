import seaborn_image as isns
import matplotlib.pyplot as plt
import os


def inp_out_ref(images, title, root):
    isns.ImageGrid(images, col_wrap=3, cbar=False, cmap='gray')
    os.mkdir(root) if not os.path.isdir(root) else None
    plt.savefig(os.path.join(root, f'{title}.jpg'), format='jpg', bbox_inches='tight', dpi=300)

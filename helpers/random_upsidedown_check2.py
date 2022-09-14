import torch
from pathlib import Path
import matplotlib.pyplot as plt
# import seaborn_image as isns
from matplotlib import image

data_path = Path("C://Users//mevan//Pictures//Legion1.jpg")
im = image.imread(data_path)
im = torch.tensor(im)
plt.imshow(im)
plt.show()

print()

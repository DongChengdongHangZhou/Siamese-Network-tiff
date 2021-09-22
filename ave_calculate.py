import tifffile as tiff
import numpy as np
from tifffile import tifffile
img = np.zeros((1,128))
for i in range(8496):
    print(i)
    img = img + tiff.imread('./1/'+str(i)+'.tiff')

img = img/8496
tiff.imwrite('ave.tiff',img)
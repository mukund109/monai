# Slow style transfer

Slow style transfer using summary statistics, as originally described by Gatys et al is still the most flexible and robust method for *multi-resolution* style transfer. However, its difficult to obtain ultra high resolution images due to memory limitations, especially when using the LBFGS optimizer. This repo is contains code for my attempts at solving this problem by by trading high memory usage for more computation time. A lot of ideas have been borrowed from the following papers:

It also contains the implementation of some original techniques that allow arbitrarily high resolution style transfer while keeping the memory usage of the GPU under control. This is done using a patch based mechanism to stylize parts of the image seperately. This is because most low level features of an ultra high resolution artwork (like the edges of brush strokes, or the texture of the canvas) don't have long range dependencies, so it makes little sense to stylize the whole image at once.

Notes/Todo :
1. VGG weights have not been normalized to give a mean activation of 1.
2. Correlation chain

Some results:

This image was created by stylizing a low resolution image (300x300), then using the output to initialize a higher resolution stylized image. This was repeated until gpu memory limits were reached. After that the patch based method implemented by the function `stylize_patches` was used.

(300,300) --> (600,600) --> (1000,1000) --> patch based (2000,2000)

Content:
![content](sample_images/aurat.jpeg)

Result:
![result](sample_images/final_output-1.jpg)

Styles:
![style](sample_images/head_of_pros_GAP.jpg)
![style2](sample_images/old_man_portrait_GAP.jpg)

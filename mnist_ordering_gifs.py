import numpy as np

from PIL import Image, ImageDraw

data = np.load("mnist_test.npy")
sample = data[7700]
# Image.fromarray(255 * sample).show()

(rows, cols) = sample.shape
buffer = 3
rand_arr = np.zeros((rows + buffer, cols, 3), dtype="uint8")
rand_arr[:rows, :cols, 1] = 255
# Image.fromarray(rand_arr).show()
fixed_arr = np.zeros((rows + buffer, cols, 3), dtype="uint8")
fixed_arr[:rows, :cols, 1] = 255

pix_idxs = np.arange(sample.size)
np.random.shuffle(pix_idxs)

rand_imgs = []
fixed_imgs = []
scale = 4
new_shape = (scale * cols, scale * (rows + buffer))
text_xy = (scale * cols // 2 - 11, scale * rows)
(prev_row, prev_col) = (None, None)
for (pix_num, pix_idx) in enumerate(pix_idxs):
    # Fill in random order array.
    if not (prev_row is None):
        rand_arr[prev_row, prev_col] = 255 * sample[prev_row, prev_col]

    row = pix_idx // cols
    col = pix_idx % cols
    rand_arr[row, col] = (255, 0, 0)
    img = Image.fromarray(rand_arr).resize(new_shape, resample=0)
    draw = ImageDraw.Draw(img)
    draw.text(text_xy, f"{100 * pix_num / sample.size:.1f}%", (255, 255, 255))
    rand_imgs.append(img)
    (prev_row, prev_col) = (row, col)

    # Fill in fixed order array.
    if pix_num > 0:
        (fixed_prev_row, fixed_prev_col) = ((pix_num - 1) // cols, (pix_num - 1) % cols)
        fixed_arr[fixed_prev_row, fixed_prev_col] = (
            255 * sample[fixed_prev_row, fixed_prev_col]
        )

    row = pix_num // cols
    col = pix_num % cols
    fixed_arr[row, col] = (255, 0, 0)
    img = Image.fromarray(fixed_arr).resize(new_shape, resample=0)
    draw = ImageDraw.Draw(img)
    draw.text(text_xy, f"{100 * pix_num / sample.size:.1f}%", (255, 255, 255))
    fixed_imgs.append(img)

rand_arr[prev_row, prev_col] = sample[prev_row, prev_col]
img = Image.fromarray(rand_arr).resize(new_shape, resample=0)
draw = ImageDraw.Draw(img)
draw.text(text_xy, f"{100 * pix_num / sample.size:.1f}%", (255, 255, 255))
rand_imgs.append(img)

(fixed_prev_row, fixed_prev_col) = ((pix_num - 1) // cols, (pix_num - 1) % cols)
fixed_arr[fixed_prev_row, fixed_prev_col] = 255 * sample[fixed_prev_row, fixed_prev_col]
img = Image.fromarray(fixed_arr).resize(new_shape, resample=0)
draw = ImageDraw.Draw(img)
draw.text(text_xy, f"{100 * pix_num / sample.size:.1f}%", (255, 255, 255))
fixed_imgs.append(img)

rand_imgs[0].save(
    fp="random_order.gif",
    format="GIF",
    append_images=rand_imgs[1:],
    save_all=True,
    duration=75,
    loop=0,
)

fixed_imgs[0].save(
    fp="fixed_order.gif",
    format="GIF",
    append_images=fixed_imgs[1:],
    save_all=True,
    duration=75,
    loop=0,
)

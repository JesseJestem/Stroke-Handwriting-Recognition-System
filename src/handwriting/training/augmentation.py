import logging

import numpy as np
from PIL import Image

#!!!!!!!!!!!!!!!!!!!!!
#Settings
#!!!!!!!!!!!!!!!!!!!!!
JITTER_STD = 0.005

ANGLE_MIN = -15.0
ANGLE_MAX = 15.0

SCALE_MIN = 0.90
SCALE_MAX = 1.10

SHIFT_MIN = -4
SHIFT_MAX = 4

logger = logging.getLogger(__name__)

#~~~~~~~~~~~~~~~~~~~~~
#Shift image without wrapping around edges
#~~~~~~~~~~~~~~~~~~~~~~

def shift_image_no_wrap(image: np.ndarray, dx: int, dy: int) -> np.ndarray:

    #----------------
    #create new image
    h, w = image.shape
    shifted = np.zeros_like(image)

    #----------------
    #where safe to take
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)

    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)

    #---------------
    #where safe to paste
    dst_x_start = max(0, dx)
    dst_x_end = min(w, w + dx)

    dst_y_start = max(0, dy)
    dst_y_end = min(h, h + dy)

    #--------------
    #maint shift
    if src_x_end > src_x_start and src_y_end > src_y_start:
        shifted[
            dst_y_start:dst_y_end,
            dst_x_start:dst_x_end,
        ] = image[
            src_y_start:src_y_end,
            src_x_start:src_x_end,
        ]

    return shifted

#~~~~~~~~~~~~~~~~~~~~~~
#Scale image 64x64 -> 64x64, black
#~~~~~~~~~~~~~~~~~~~~~~

def scale_image(image: np.ndarray, scale: float) -> np.ndarray:

    size = image.shape[0]

    #---------------
    #form 0.0-1.0 to 0-255
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8)

    new_size = max(1, int(size * scale))

    #---------------
    #resize img, BILINEAR - smooth
    resized = pil_image.resize((new_size, new_size), resample=Image.Resampling.BILINEAR)

    #---------------
    #crete new black img 'L' - grayscale
    result = Image.new("L", (size, size), 0)

    #-------------
    #return size of image to 64x64
    if new_size >= size:
        left = (new_size - size) // 2
        top = (new_size - size) // 2
        cropped = resized.crop((left, top, left + size, top + size))
        result.paste(cropped, (0, 0))

    #------------
    #if smaller just paste in center of black img
    else:
        left = (size - new_size) // 2
        top = (size - new_size) // 2
        result.paste(resized, (left, top))

    #-----------------
    #return to np array: 0 ... 255 -> 0.0 ... 1.0
    result_array = np.array(result).astype(np.float32) / 255.0

    return result_array

#~~~~~~~~~~~~~~~~~~~~~~
#Full img augment: scale → rotate → shift
#~~~~~~~~~~~~~~~~~~~~~~

def augment_image(
        image: np.ndarray,
        angle: float,
        scale: float,
        dx_pixels: int,
        dy_pixels: int,
) -> np.ndarray:
    
    #-----------------------------
    #img to 2d(64, 64, 1) - > (64, 64, 1)
    image_2d = image.squeeze()

    #scaling
    image_2d = scale_image(image_2d, scale)

    #---------------------
    #rotate prepare
    image_uint8 = np.clip(image_2d * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8)

    rotated = pil_image.rotate(
        angle,
        #set px smooth by near px
        resample=Image.Resampling.BILINEAR,
        fillcolor=0
    )

    rotated_array = np.array(rotated).astype(np.float32) / 255.0

    shifted = shift_image_no_wrap(
        rotated_array,
        dx=dx_pixels,
        dy=dy_pixels,
    )

    shifted = np.clip(shifted, 0.0, 1.0)

    #(64, 64) -> (64, 64, 1)
    #... - take all row&cols and add new demention
    return shifted[..., np.newaxis].astype(np.float32)

#~~~~~~~~~~~~~~~~~~~~~~
#Full stroke augment: same as img
#~~~~~~~~~~~~~~~~~~~~~~

def augment_strokes(
        strokes: np.ndarray,
        angle: float,
        scale: float,
        dx_pixels: int,
        dy_pixels: int,
        jitter_std: float = 0.005,
) -> np.ndarray:
    
    augmented = strokes.copy().astype(np.float32)

    #check wich values is close to 0 on each dimention, ~ - logic negative
    active_mask = ~np.all(np.isclose(augmented, 0.0), axis=1)

    #if dont have points -> return
    if not np.any(active_mask):
        return augmented
    
    #take all x, y of real points
    coords = augmented[active_mask, :2]

    #-----------------------
    #move center to 0, 0
    coords = coords - 0.5

    #-----------------
    #rotation
    #change angle to radians
    radians = np.deg2rad(angle)
    cos_value = np.cos(radians)
    sin_value = np.sin(radians)

    #count new x, y
    #x_new = x * cos(θ) - y * sin(θ)
    #y_new = x * sin(θ) + y * cos(θ)
    rotation_matrix = np.array(
        [
            [cos_value, sin_value],
            [-sin_value, cos_value]
        ],
        dtype=np.float32,
    )

    #rotate all points
    coords = coords @ rotation_matrix.T

    #----------------
    #scale
    coords = coords * scale
    
    #return center
    coords = coords + 0.5

    #--------------------
    #shift in pixels
    dx_norm = dx_pixels / 64.0
    dy_norm = dy_pixels / 64.0

    coords[:, 0] += dx_norm #shift x
    coords[:, 1] += dy_norm #shift y

    #return coords back to array copy
    augmented[active_mask, :2] = coords

    #---------------------------
    #Add noize to drawing points where pen_down in >= 0.5
    drawing_mask = augmented[:, 4] >= 0.5

    #normal distribution for noize
    noise = np.random.normal(
        loc=0.0, #average noize value
        scale=jitter_std, #power of noize
        size=augmented[:, :2].shape, #size of noizy array
    ).astype(np.float32)

    augmented[drawing_mask, :2] += noise[drawing_mask]

    #-----------
    #clip x,y if was changed
    augmented[:, 0] = np.clip(augmented[:, 0], 0.0, 1.0)
    augmented[:, 1] = np.clip(augmented[:, 1], 0.0, 1.0)

    #clip t and pressure
    augmented[:, 2] = np.clip(augmented[:, 2], 0.0, 1.0)
    augmented[:, 3] = np.clip(augmented[:, 3], 0.0, 1.0)

    #save binary feature 0 or 1
    augmented[:, 4] = (augmented[:, 4] >= 0.5).astype(np.float32)
    augmented[:, 5] = (augmented[:, 5] >= 0.5).astype(np.float32)

    return augmented.astype(np.float32)

#~~~~~~~~~~~~~~~~~~~~~~
#Augment pairs img & stroke with same geometric
#~~~~~~~~~~~~~~~~~~~~~~

def augment_pair(
        image: np.ndarray,
        strokes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    
    #----------------------
    #random augmentation
    angle = np.random.uniform(ANGLE_MIN, ANGLE_MAX)
    scale = np.random.uniform(SCALE_MIN, SCALE_MAX)

    dx_pixels = np.random.randint(SHIFT_MIN, SHIFT_MAX + 1)
    dy_pixels = np.random.randint(SHIFT_MIN, SHIFT_MAX + 1)

    #------------------
    #augment image
    augmented_image = augment_image(
        image=image,
        angle=angle,
        scale=scale,
        dx_pixels=dx_pixels,
        dy_pixels=dy_pixels
    )

    #----------------
    #augment stroke
    augmented_strokes = augment_strokes(
        strokes=strokes,
        angle=angle,
        scale=scale,
        dx_pixels=dx_pixels,
        dy_pixels=dy_pixels,
        jitter_std =JITTER_STD,
    )

    return augmented_image, augmented_strokes

#~~~~~~~~~~~~~~~~~~~~~~
#Augment all training set by default = 1
#~~~~~~~~~~~~~~~~~~~~~~

def augment_training_data(
        X_train_images: np.ndarray,
        X_train_strokes: np.ndarray,
        y_train: np.ndarray,
        copies_per_sample: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    #if no need augmentation - return
    if copies_per_sample <= 0:
        return X_train_images, X_train_strokes, y_train
    
    #save originals
    images_list = [X_train_images]
    strokes_list = [X_train_strokes]
    labels_list = [y_train]

    #container for new data
    augmented_images = []
    augmented_strokes = []
    augmented_labels = []

    #--------------------
    #augmentation of training set
    for copy_index in range(copies_per_sample):
        logger.info(
            "Creating augmented copy %d/%d",
            copy_index + 1,
            copies_per_sample,
        )

        for image, strokes, label in zip(X_train_images, X_train_strokes, y_train):
            aug_image, aug_strokes = augment_pair(image, strokes)

            #add to list
            augmented_images.append(aug_image)
            augmented_strokes.append(aug_strokes)
            augmented_labels.append(label)

    #transform list to np array
    augmented_images_array = np.array(augmented_images, dtype=np.float32)
    augmented_strokes_array = np.array(augmented_strokes, dtype=np.float32)
    augmented_labels_array = np.array(augmented_labels, dtype=y_train.dtype)

    #add new augmented data to originals
    images_list.append(augmented_images_array)
    strokes_list.append(augmented_strokes_array)
    labels_list.append(augmented_labels_array)

    #add data to training set to the bottom, along the num of samples
    X_train_images_aug = np.concatenate(images_list, axis=0)
    X_train_strokes_aug = np.concatenate(strokes_list, axis=0)
    y_train_aug = np.concatenate(labels_list, axis=0)

    #shuffle all examples
    indices = np.random.permutation(len(y_train_aug))
    X_train_images_aug = X_train_images_aug[indices]
    X_train_strokes_aug = X_train_strokes_aug[indices]
    y_train_aug = y_train_aug[indices]

    return X_train_images_aug, X_train_strokes_aug, y_train_aug

#~~~~~~~~~~~~~~~~~~~~~
#TEST BLOCK
#~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = BASE_DIR / "data" / "processed" / "dataset.npz"

    data = np.load(DATASET_PATH, allow_pickle=True)

    X_images = data["X_images"]
    X_strokes = data["X_strokes"]
    y = data["y"]
    display_labels = data["display_labels"]

    index = 650

    image = X_images[index]
    strokes = X_strokes[index]
    label = display_labels[y[index]]

    aug_image, aug_strokes = augment_pair(image, strokes)

    plt.figure()
    plt.title(f"Original image: {label}")
    plt.imshow(image.squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()

    plt.figure()
    plt.title(f"Augmented image: {label}")
    plt.imshow(aug_image.squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()

    plt.figure()
    plt.title("Original strokes")
    drawing = strokes[strokes[:, 4] >= 0.5]
    plt.scatter(drawing[:, 0], drawing[:, 1], s=10)
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.title("Augmented strokes")
    drawing = aug_strokes[aug_strokes[:, 4] >= 0.5]
    plt.scatter(drawing[:, 0], drawing[:, 1], s=10)
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.grid(True)
    plt.show()
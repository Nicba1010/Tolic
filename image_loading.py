import tensorflow as tf

vocabulary = ['2', '3', '4', '5', '6', '7', '8', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']

char_to_num = {x: i for i, x in enumerate(vocabulary)}

offset = int((276 - 200) / 2)

def encode_single_sample(img_path, label, crop):
    # Read image file and returns a tensor with dtype=string
    img = tf.io.read_file(img_path)
    # Decode and convert to grayscale (this conversion does not cause any information lost and reduces the size of the tensor)
    # This decode function returns a tensor with dtype=uint8
    img = tf.io.decode_png(img, channels=1)
    # Scales and returns a tensor with dtype=float32
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Remove first and last 13 pictures to crop into nice size
    img = tf.image.crop_to_bounding_box(img, offset_height=0, offset_width=int((276 - 200) / 2), target_height=50,
                                        target_width=200)
    # Crop and resize to the original size :
    # top-left corner = offset_height, offset_width in image = 0, 25
    # lower-right corner is at offset_height + target_height, offset_width + target_width = 50, 150
    if (crop == True):
        img = tf.image.crop_to_bounding_box(img, offset_height=0, offset_width=(80 - offset), target_height=50,
                                            target_width=120)
        img = tf.image.resize(img, size=[50, 200], method='bilinear', preserve_aspect_ratio=False, antialias=False,
                              name=None)
    # Transpose the image because we want the time dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # Converts the string label into an array with 5 integers. E.g. '6n6gg' is converted into [6,16,6,14,14]
    label = list(map(lambda x: char_to_num[x], label))
    return img.numpy(), label

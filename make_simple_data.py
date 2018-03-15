import gzip, pickle, os.path
import numpy as np
import struct

# Generate batch files to be loaded by Coq/OCaml.

# Number of bits (e.g., 16 or 32) per floating-point parameter
N = 16
BATCH_SIZE = 100

with open('emnist/all.pkl', 'rb') as f:
# with open('emnist/test.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

images = data.images
labels = data.labels

print(images.shape)

# This function stolen from code posted to:
# https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
def binary(num):
  return ''.join(bin(c).replace('0b', '').rjust(8, '0')
                 for c in struct.pack('!f', num))
# END stolen

def float_cast(f):
    if N == 32: return np.float32(f)
    elif N == 16: return np.float16(f)
    else: return f

# Indices record the '1' bits.
def float_to_bin(f):
    b = binary(float_cast(f).item())
    l = zip(list(range(N)), [i for i in b])
    # Just the nonzero indices
    return list(map(lambda p: p[0], filter(lambda x: x[1] == '1', l)))

def encode_image(image):
    return list(filter(lambda p: p[1],
                       [(i, float_to_bin(image[i]))
                   for i in range(image.shape[0])]))

os.makedirs('basic_data', exist_ok=True)

for i in range(0, images.shape[0], BATCH_SIZE):
    batch_images = images[i:i+BATCH_SIZE,:]
    batch_labels = labels[i:i+BATCH_SIZE]
    encoded_images = list(map(encode_image, batch_images))
    with open('basic_data/batch_' + str(i//BATCH_SIZE), 'w') as f:
        for j in range(BATCH_SIZE):
            encoded_image = encoded_images[j]
            f.write(str(batch_labels[j]) + ' ')
            for (x, bits) in encoded_image:
                f.write(str(x) + ' ' + str(len(bits)) + ' ' + \
                        ' '.join(map(str, bits)) + ' ')
            f.write('\n')


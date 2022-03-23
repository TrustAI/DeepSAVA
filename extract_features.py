"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np
import os.path
from data import DataSet
from extractor import Extractor
from tqdm import tqdm
import PIL
import argparse
# Set defaults.
seq_length = 40
class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.
parser = argparse.ArgumentParser(description='Use Adam optimizer to generate adversarial examples.')
parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory of dataset.')
parser.set_defaults(use_crop=True)
args = parser.parse_args()
# Get the dataset.
data = DataSet(path = args.input_dir,seq_length=seq_length, class_limit=class_limit)

# get the model.

model = Extractor()

# Loop through data.
pbar = tqdm(total=len(data.data))
for video in data.data:

    # Get the path to the sequence for this video.
    path = os.path.join('data', 'sequences', video[1]+'-'+video[2] + '-' + str(seq_length) + \
        '-features')  # numpy will auto-append .npy
    # Check if we already have it.
    if os.path.isfile(path + '.npy'):
        pbar.update(1)
        continue

    # Get the frames for this video.
    frames = data.get_frames_for_sample(args.input_dir,video)
    if len(frames)<seq_length:
     #   print(frames)

        continue    
    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in frames:
        try:
            img = PIL.Image.open(image)
        except PIL.UnidentifiedImageError:
            print(image)
            os.remove(image)
            continue
        features = model.extract(image)
        #if features is not  None: 
        sequence.append(features)
        #else:
         #   pass
    # Save the sequence.
    np.save(path, sequence)

    pbar.update(1)

pbar.close()

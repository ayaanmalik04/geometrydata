from src.mrcnn import visualize
from DataGenerator import generate_geometry_episodes
import os
import numpy as np
from PIL import Image
class Args:
    def __init__(self):
        self.mask_size = 5
        self.history_size = 1
        self.generate_levels = ".*"
        self.train_epochs = 2
        self.val_epochs = 1
        self.visualize = 1

args = Args()

data_gen, data_gen_val = generate_geometry_episodes(args)

# output_folder = "generated_images"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

for image_id in range(data_gen.last_id):
    image = data_gen.load_image(image_id)
    masks, class_ids = data_gen.load_mask(image_id)

    # filename = os.path.join(output_folder, "image_{:04d}.png".format(image_id))C
    # image_pil = Image.fromarray(image)
    # image_pil.save(filename)
    # visualize.display_top_masks(image, masks, class_ids, list(data_gen.id_name_dic.keys()))
    visualize.display_images([image])

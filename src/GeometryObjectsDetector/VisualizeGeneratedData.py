from src.mrcnn import visualize
from DataGenerator import generate_geometry_episodes
import os
from PIL import Image
class Args:
    def __init__(self):
        self.mask_size = 5
        self.history_size = 1
        self.generate_levels = ".*"
        self.train_epochs = 1000
        self.val_epochs = 100
        self.visualize = 0

args = Args()

data_gen, data_gen_val = generate_geometry_episodes(args)
# Create a folder to store the images
output_folder = "generated_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_id in range(data_gen.last_id):
    image = data_gen.load_image(image_id)
    masks, class_ids = data_gen.load_mask(image_id)

    # Save the image to the output folder with a unique filename based on the image ID
    filename = os.path.join(output_folder, "image_{:04d}.png".format(image_id))
    image_pil = Image.fromarray(image)
    image_pil.save(filename)

    # You can also display the image if needed
    # visualize.display_images([image], titles=["Image ID: {}".format(image_id)], cols=1)

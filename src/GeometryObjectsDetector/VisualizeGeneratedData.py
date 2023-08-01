from src.mrcnn import visualize
from DataGenerator import generate_geometry_episodes

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

for image_id in range(data_gen.last_id):
    image = data_gen.load_image(image_id)
    masks, class_ids = data_gen.load_mask(image_id)
    #visualize.display_top_masks(image, masks, class_ids, list(data_gen_val.id_name_dic.keys()))
    visualize.display_images([image], titles=["Image ID: {}".format(image_id)], cols=1)


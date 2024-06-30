import os


class ImageRenamer:
    def __init__(self, directory, extension):
        self.directory = directory
        self.extension = extension

    def rename_images(self):
        images = [f for f in os.listdir(self.directory) if f.endswith(self.extension)]
        images.sort()
        for i, image in enumerate(images, start=1):
            new_name = f"{i}{self.extension}"
            os.rename(os.path.join(self.directory, image), os.path.join(self.directory, new_name))
        print(f"Renamed {len(images)} images.")


directory = '../../datasets/heineiken-datasets'
extension = '.jpg'
renamer = ImageRenamer(directory, extension)
renamer.rename_images()

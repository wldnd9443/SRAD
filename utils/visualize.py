from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

def visualize_tensor(tensor):
    plt.figure()
    plt.imshow(to_pil_image(tensor), cmap='gray')
    plt.show()

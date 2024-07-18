import matplotlib.pyplot as plt
import numpy as np

# Define a function to create a flowchart
def create_flowchart():
    fig, ax = plt.subplots(figsize=(15, 10))

    # Define the blocks of the workflow
    blocks = {
        'input': {'pos': (0.5, 5), 'text': 'Input: Confocal Microscopy Image'},
        'preprocessing': {'pos': (0.5, 4), 'text': 'Preprocessing:\nResolution, Z-stack, Bit depth\nNuclear area detection\nFiber orientation'},
        'segmentation': {'pos': (0.5, 3), 'text': 'Image Segmentation:\nU-Net CNN Architecture\nManual thresholding'},
        'reconstruction': {'pos': (0.5, 2), 'text': 'Reconstruction:\nConnect actin dots\nReconstruct nucleus'},
        'output': {'pos': (0.5, 1), 'text': 'Output: Measurements\nand Reconstruction'}
    }

    # Draw blocks
    for block in blocks.values():
        rect = plt.Rectangle(block['pos'], 1, 0.8, fc='skyblue')
        ax.add_patch(rect)
        plt.text(block['pos'][0]+0.5, block['pos'][1]+0.4, block['text'], ha='center', va='center')

    # Draw arrows
    for i in range(len(blocks)-1):
        block1 = list(blocks.keys())[i]
        block2 = list(blocks.keys())[i+1]
        ax.annotate('', xy=blocks[block2]['pos'] + np.array([0.5, 0.8]), xytext=blocks[block1]['pos'] + np.array([0.5, 0]),
                    arrowprops=dict(arrowstyle="->", color='black'))

    # Set chart properties
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 6)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('flow_chart.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    return 'flow_chart.png'

# Create the flowchart and retrieve the saved path
flowchart_path = create_flowchart()
flowchart_path

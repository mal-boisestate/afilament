import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os

def extract_scalar_data(log_file):
    ea = event_accumulator.EventAccumulator(log_file, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = [(event.step, event.value) for event in events]
    return data

# Specify your log directory
log_dir = r'D:\BioLab\scr_2.0\unet\runs\Aug24_14-09-54_LAPTOP-ISSFJ09KLR_0.001_BS_1_SCALE_1.0'

# Initialize data structures
training_loss = []
validation_loss = []
training_dice = []
validation_dice = []

# Process each log file
for root, dirs, files in os.walk(log_dir):
    for file in files:
        if 'events.out.tfevents' in file:
            log_file_path = os.path.join(root, file)
            scalar_data = extract_scalar_data(log_file_path)

            # Extract the data
            for tag, values in scalar_data.items():
                if tag == 'Loss/Train':
                    training_loss.extend(values)
                elif tag == 'Loss/Val':
                    validation_loss.extend(values)
                elif tag == 'Dice/Train':
                    training_dice.extend(values)
                elif tag == 'Dice/Val':
                    validation_dice.extend(values)

# Sort the values by step (epoch)
training_loss.sort(key=lambda x: x[0])
validation_loss.sort(key=lambda x: x[0])
training_dice.sort(key=lambda x: x[0])
validation_dice.sort(key=lambda x: x[0])

# Extract steps and values for plotting
train_loss_steps, train_loss_vals = zip(*training_loss)
val_loss_steps, val_loss_vals = zip(*validation_loss)
train_dice_steps, train_dice_vals = zip(*training_dice)
val_dice_steps, val_dice_vals = zip(*validation_dice)

# Create plots
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_loss_steps, train_loss_vals, label='Training Loss')
plt.plot(val_loss_steps, val_loss_vals, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

# Dice plot
plt.subplot(1, 2, 2)
plt.plot(train_dice_steps, train_dice_vals, label='Training Dice')
plt.plot(val_dice_steps, val_dice_vals, label='Test Dice')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.title('Training and Test Dice Coefficient')
plt.legend()

plt.tight_layout()
plt.show()

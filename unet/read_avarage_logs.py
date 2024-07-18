import matplotlib.pyplot as plt
import numpy as np
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

def average_logs(log_folder_path):
    log_dirs = [os.path.join(log_folder_path, d) for d in os.listdir(log_folder_path) if
                os.path.isdir(os.path.join(log_folder_path, d))]

    all_training_loss = []
    all_validation_loss = []
    all_training_dice = []
    all_validation_dice = []

    for log_dir in log_dirs:
        training_loss = []
        validation_loss = []
        training_dice = []
        validation_dice = []

        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if 'events.out.tfevents' in file:
                    log_file_path = os.path.join(root, file)
                    scalar_data = extract_scalar_data(log_file_path)

                    for tag, values in scalar_data.items():
                        if tag == 'Loss/Train':
                            training_loss.extend(values)
                        elif tag == 'Loss/Val':
                            validation_loss.extend(values)
                        elif tag == 'Dice/Train':
                            training_dice.extend(values)
                        elif tag == 'Dice/Val':
                            validation_dice.extend(values)

        all_training_loss.append(training_loss)
        all_validation_loss.append(validation_loss)
        all_training_dice.append(training_dice)
        all_validation_dice.append(validation_dice)

    def average_and_std_data(data):
        # Flatten the list of logs and sort by step
        flattened_data = [item for sublist in data for item in sublist]
        flattened_data.sort(key=lambda x: x[0])

        # Group data by step
        grouped_data = {}
        for step, value in flattened_data:
            if step not in grouped_data:
                grouped_data[step] = []
            grouped_data[step].append(value)

        # Calculate the average and standard deviation for each step
        avg_data = []
        std_data = []
        for step in grouped_data:
            values = grouped_data[step]
            avg_data.append((step, np.mean(values)))
            std_data.append(np.std(values))

        return avg_data, std_data

    avg_training_loss, std_training_loss = average_and_std_data(all_training_loss)
    avg_validation_loss, std_validation_loss = average_and_std_data(all_validation_loss)
    avg_training_dice, std_training_dice = average_and_std_data(all_training_dice)
    avg_validation_dice, std_validation_dice = average_and_std_data(all_validation_dice)

    # Plot the averaged data with transparent standard deviation areas
    plt.figure(figsize=(12, 6))

    # Training loss plot with std deviation
    plt.subplot(1, 2, 1)
    train_steps, train_values = zip(*avg_training_loss)
    train_std_values = std_training_loss
    plt.fill_between(train_steps, np.array(train_values) - np.array(train_std_values),
                     np.array(train_values) + np.array(train_std_values), alpha=0.2, color='blue')
    plt.plot(train_steps, train_values, color='blue')
    plt.ylim(0, 0.3)  # Set y-axis to start from 0.3

    # Validation loss plot with std deviation
    val_steps, val_values = zip(*avg_validation_loss)
    val_std_values = std_validation_loss
    plt.fill_between(val_steps, np.array(val_values) - np.array(val_std_values),
                     np.array(val_values) + np.array(val_std_values), alpha=0.2, color='red')
    plt.plot(val_steps, val_values, color='red')
    plt.ylim(0, 0.3)  # Set y-axis to start from 0.3

    plt.title('Average Training and Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    # Dice coefficient plot with std deviation
    plt.subplot(1, 2, 2)
    train_steps, train_values = zip(*avg_training_dice)
    train_std_values = std_training_dice
    plt.fill_between(train_steps, np.array(train_values) - np.array(train_std_values),
                     np.array(train_values) + np.array(train_std_values), alpha=0.2, color='blue')
    plt.plot(train_steps, train_values, color='blue')
    # plt.ylim(0.2)  # Set y-axis to start from 0.2

    val_steps, val_values = zip(*avg_validation_dice)
    val_std_values = std_validation_dice
    plt.fill_between(val_steps, np.array(val_values) - np.array(val_std_values),
                     np.array(val_values) + np.array(val_std_values), alpha=0.2, color='red')
    plt.plot(val_steps, val_values, color='red')
    # plt.ylim(0, 0.3)  # Set y-axis to start from 0.2

    plt.title('Average Training and Validation Dice')
    plt.xlabel('Step')
    plt.ylabel('Dice Coefficient')

    plt.tight_layout()
    plt.show()

log_folder_path = r'D:\BioLab\Current_experiments\afilament\2023.12.23_Training_paper_verification\paper2\Actin20\logs'
average_logs(log_folder_path)

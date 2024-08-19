import random
import datetime
import os
import json
import torch
import telegram
import emoji
import numpy as np
from PIL import Image
from natsort import natsorted
import yaml


def set_cwd():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)


def get_time_stamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def get_hrs_min_sec(total_seconds):
    hours = total_seconds//3600
    minutes = (total_seconds % 3600) // 60
    seconds = (total_seconds % 3600) % 60
    return int(hours), int(minutes), int(seconds)


def check_and_create_folder(path):
    """Check if a folder in given path exists, if not then create it"""
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else: return False


def print_elements_of_list(list):
    """Prints each element of list in a newline"""
    [print(element) for element in list]


def write_list_to_text_file(save_path, text_list, verbose=True):
    """
    Function to write a list to a text file.
    Each element of the list is written to a new line.
    Note: Existing text in the file will be overwritten!
    :param save_path: Path to save-should be complete with .txt extension)
    :param text_list: List of text-each elem of list written in new line)
    :param verbose: If true, prints success message to console
    :return: No return, writes file to disk and prints a success message
    """
    with open(save_path, 'w+') as write_file:
        for text in text_list:
            if isinstance(text, str):
                write_file.write(text + '\n')
            else:
                write_file.write(str(text) + '\n')
        write_file.close()
    if verbose: print("Text file successfully written to disk at {}".format(save_path))


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as file:
        with Image.open(file) as image:
            return image.convert('RGB')


def mask_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as file:
        with Image.open(file) as image:
            return image.convert('L')


def mask_np_loader(path):
    # The np mask has 3 channels but we need only a single channel
    return np.load(path)[..., 0]


def json_loader(path):
    with open(path, 'rb') as file:
        return json.load(file)


def write_to_json_file(content, path):
    with open(path, 'w') as file:
        json.dump(content, file, indent=4)


def write_to_yaml_file(content, path):
    with open(path, 'w') as file:
        yaml.safe_dump(content, file, indent=4)


def append_yaml_file(yaml_new_dict, path):
    with open(path, 'r') as yamlfile:
        yaml_dict = yaml.load(yamlfile)
        yaml_dict.update(yaml_new_dict)
    if yaml_dict:
        with open(path, 'w') as yamlfile:
            yaml.dump(yaml_dict, yamlfile)


def read_lines_from_text_file(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def convert_to_numpy_image(image_tensor):
    return image_tensor.numpy().transpose((1, 2, 0))


def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sub_dirs(path, sort=True, paths=True):
    """ Returns all the sub-directories in the given path as a list
    If paths flag is set, returns the whole path, else returns just the names
    """
    sub_things = os.listdir(path)  # thing can be a folder or a file
    if sort: sub_things = natsorted(sub_things)
    sub_paths = [os.path.join(path, thing) for thing in sub_things]

    sub_dir_paths = [sub_path for sub_path in sub_paths if os.path.isdir(sub_path)]  # choose only sub-dirs
    sub_dir_names = [os.path.basename(sub_dir_path) for sub_dir_path in sub_dir_paths]

    return sub_dir_paths if paths else sub_dir_names


def send_telegram_message_to_lalith(text='..'):
    """Send a message to the Bot
    """
    bot = telegram.Bot(token='1226613669:AAFj4tztWE2VvOkanja8dyLkYPSCkbWaOSQ')
    bot.send_message(chat_id=424150566, text=text)


def send_training_completion_message(name, num_epochs):
    """Prepare a message and send it to a telegram bot
    """
    message = emoji.emojize('Hi Lalith, the training for the experiment "{}" for the suture detection \
    model has completed training for {} epochs :thumbs_up:'.format(name,
                                                                   num_epochs))
    send_telegram_message_to_lalith(message)

def get_last_non_zero_elem(list_):
    # Get the previous non-zero frame in the sequence
    # If the last element of the list is non zero or IS the last element of the whole list then base condition is reached
    # Else it checks again with one element less
    non_zero_elem = list_[-1] if list_[-1] or len(list_)==1 else get_last_non_zero_elem(list_[:-1])
    return non_zero_elem

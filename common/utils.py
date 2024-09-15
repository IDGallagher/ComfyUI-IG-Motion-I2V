import hashlib
import os
from typing import Iterable
import re

def get_sorted_dir_files_from_directory(directory: str, skip_first_images: int=0, select_every_nth: int=1, extensions: Iterable=None):
    directory = directory.strip()
    dir_files = os.listdir(directory)
    dir_files = sorted(dir_files)
    dir_files = [os.path.join(directory, x) for x in dir_files]
    dir_files = list(filter(lambda filepath: os.path.isfile(filepath), dir_files))
    # filter by extension, if needed
    if extensions is not None:
        extensions = list(extensions)
        new_dir_files = []
        for filepath in dir_files:
            ext = "." + filepath.split(".")[-1]
            if ext.lower() in extensions:
                new_dir_files.append(filepath)
        dir_files = new_dir_files
    # start at skip_first_images
    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    return dir_files

# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    h = hashlib.sha256()
    b = bytearray(10*1024*1024) # read 10 megabytes at a time
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        i = 0
        # don't hash entire file, only portions of it if requested
        while n := f.readinto(mv):
            if i%hash_every_n == 0:
                h.update(mv[:n])
            i += 1
    return h.hexdigest()

def rename_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Replace only whole words to avoid partial replacements
        new_key = re.sub(r'\bquery\b', 'to_q', new_key)
        new_key = re.sub(r'\bkey\b', 'to_k', new_key)
        new_key = re.sub(r'\bvalue\b', 'to_v', new_key)
        new_key = re.sub(r'\bproj_attn\b', 'to_out.0', new_key)
        new_state_dict[new_key] = value
    return new_state_dict

def print_loading_issues(missing_keys, unexpected_keys):
    if missing_keys:
        print(f"Missing keys when loading state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys when loading state_dict: {unexpected_keys}")


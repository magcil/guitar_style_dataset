import os
import wave
import logging
import contextlib
import subprocess

CLASS_MAPPING = {
    0: 'alternate picking',
    1: 'legato',
    2: 'tapping',
    3: 'sweep picking',
    4: 'vibrato',
    5: 'hammer on',
    6: 'pull off',
    7: 'slide',
    8: 'bend'
}

def crawl_directory(directory: str, extension: str = None) -> list:
    """Crawling data directory
    Args:
        directory (str) : The directory to crawl
    Returns:
        tree (list)     : A list with all the filepaths
    """
    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            if extension is not None:
                if _file.endswith(extension):
                    tree.append(os.path.join(subdir, _file))
            else:
                tree.append(os.path.join(subdir, _file))
    return tree

def get_label(filename:str) -> str:
    prev_str = 'class_'
    idx = filename.rfind(prev_str)
    idx += len(prev_str)
    label = filename[idx]
    return label

def get_wav_duration(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration
    
def prepare_dirs(input_dir, train_wavs, test_wavs, output_path, segment_size):
    """Given a train/test split create dirs with segmented wavs on train separated in classes"""
    train_path = os.path.join(output_path, 'train')
    test_path = os.path.join(output_path, 'test')
    temp_wav_path = os.path.join(input_dir, "temp.wav")
    input_files = os.listdir(input_dir)
    os.mkdir(train_path)
    os.mkdir(test_path)
    train_path_dict = {}
    for label, guitar_technique in CLASS_MAPPING.items():
        tmp_train_path = os.path.join(train_path, guitar_technique)
        os.mkdir(tmp_train_path)
        train_path_dict[label] = tmp_train_path
    for wav_file in input_files:
        wav_name = wav_file.rstrip(".wav")
        wav_path = os.path.join(input_dir, wav_file)
        if wav_file in train_wavs:
            label = get_label(wav_file)
            out_path = train_path_dict[label]
        elif wav_file in test_wavs:
            out_path = test_path
        else:
            print(f'File {wav_file} does not belong to train nor test set. \nSkipping {wav_file}.')
        # get wav duration
        if 'out_path' in locals():
            try:
                dur = get_wav_duration(os.path.join(input_dir, wav_file))
            except Exception as err:
                raise err
        else:
            pass
        # create a temporary wav that has been trimmed accordingly depending on the segment size
        if 'dur' in locals():
            end = (dur // segment_size) * segment_size
            try:
                subprocess.check_call(
                    [
                        "ffmpeg", "-i", wav_path, "-ss", "0", "-to",
                        str(end), "-c", "copy", "-y", "-loglevel", "quiet", temp_wav_path
                    ]
                )
            except subprocess.CalledProcessError as e:
                print(f"An error occured saving the temp wav of {wav_file}.\nError: {e}")
                print(f"Skipping the splitting of {wav_file}.")
                continue
            # segment temporary wav and save it in corresponding directory
            try:
                subprocess.check_call(
                    [
                        "ffmpeg", "-i", temp_wav_path, "-f", "segment", "-segment_time",
                        str(segment_size), "-ar", "8000", "-ac", "1", "-c", "copy", "-loglevel", "quiet",
                        f"{out_path}/{wav_name}_{segment_size}_%03d.wav"
                    ]
                    )
            except subprocess.CalledProcessError as e:
                print(f"An error occured with the segmentation of {wav_file}.\nError: {e}")
        else:
            pass


def deep_audio_training():
    """Train on dirs using deep audio features"""
    pass


def validate_on_test():
    """Validate on test using deep audio features"""
    pass
import os

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

def prepare_dirs(train_wavs, test_wavs, output_path, segment_size):
    """Given a train/test split create dirs with segmented wavs on train separated in classes"""
    train_path = os.path.join(output_path, 'train')
    test_path = os.path.join(output_path, 'test')

    # Prepare test dirs
    for label, guitar_technique in CLASS_MAPPING.items():
        
        pass
    pass


def deep_audio_training():
    """Train on dirs using deep audio features"""
    pass


def validate_on_test():
    """Validate on test using deep audio features"""
    pass
import argparse
import os
import sys
from pathlib import Path
import yaml
from tqdm import tqdm

from utils.dataloaders import IMG_FORMATS

sys.path.append(str(Path(__file__).parent.joinpath('lmdb')))
from lmdbDataset import *

def replace_last_occurrences(s, old, new, count=1):
    li = s.rsplit(old, count)
    return new.join(li)

def img2label(path):
    path_str = path.absolute().as_posix()
    suffix = path.suffix
    if suffix:
        path_str = replace_last_occurrences(path_str, suffix, '.txt')
    if path_str.endswith('/images'):
        return replace_last_occurrences(path_str, 'images', 'labels')
    if '/images/' not in path_str:
        raise ValueError
    return Path(replace_last_occurrences(path_str, '/images/', '/labels/'))

parser = argparse.ArgumentParser(description='Convert a YOLO dataset to an LMDB.')
parser.add_argument('yaml_path', help="the path to the dataset's YAML file")
parser.add_argument('lmdb_path', help="the path where the lmdb dataset will be created (must be empty)")
parser.add_argument('--source', default='', help="a string describing the dataset's source")
parser.add_argument('--ignore-empty', action='store_true')
parser.add_argument('--overwrite', action='store_true')

args = parser.parse_args()

if args.overwrite:
    args.ignore_empty = True

lmdb_path = Path(args.lmdb_path)
if not lmdb_path.exists():
    try:
        lmdb_path.mkdir(parents=True)
    except:
        print(f'ERROR: the LMDB destination directory does not exist and could not be created, exiting.')
        exit(-1)
elif not lmdb_path.is_dir():
    print(f'ERROR: the LMDB destination path is not a directory, exiting.')
    exit(-1)
elif not args.ignore_empty and list(lmdb_path.glob('*')):
    print(f'ERROR: the LMDB destination path is not an empty directory, exiting.')
    exit(-1)

yaml_path = Path(args.yaml_path)
if not yaml_path.is_file():
    print('ERROR: specified YAML path does not point to a file, exiting.')
    exit(-1)

with yaml_path.open() as yaml_file:
    try:
        yaml_dict = yaml.safe_load(yaml_file)
    except:
        print('ERROR: could not parse YAML file, exiting.')
        exit(-1)

if yaml_dict.get('lmdb', False):
    print('YAML files already has the "lmdb" field set to true, the dataset may already be in LMDB format, exiting.')
    exit(-1)

req_fields = ['nc']
for field in req_fields:
    if field not in yaml_dict:
        print(f'ERROR: YAML file does not contain required field "{field}", exiting.')
        exit(-1)
sets = []
for set_ in ['train', 'val', 'test']:
    if set_ in yaml_dict:
        sets.append(set_)

try:
    nc = int(yaml_dict['nc'])
except:
    print('The "nc" field must be an integer')
    exit(-1)

imgs_paths = {}
for set_ in sets:
    try:
        p = Path(yaml_dict[set_])
        if not p.is_absolute():
            p = yaml_path.parent.joinpath(p)
        imgs_paths[set_] = p
    except:
        print(f'ERROR: could not parse the {set_} images path, exiting.')
        exit(-1)
    if not imgs_paths[set_].is_dir():
        print(f'ERROR: the {set_} images path is not a directory, exiting.')
        exit(-1)

for set_ in sets:
    img_paths = list(imgs_paths[set_].glob('*'))
    warnings = []
    with LmdbDataset(str(lmdb_path.joinpath(set_).absolute().as_posix())) as dataset:
        for img_path in tqdm(img_paths, desc=f'{set_} set progress', ):
            if img_path.is_dir():
                continue
            if not img_path.suffix[1:] in IMG_FORMATS:
                continue

            with img_path.open(mode='rb') as fin: image_data = fin.read()
            
            label_path = img2label(img_path)
            with label_path.open() as fl:
                data = fl.readlines()
            
            boxes = []

            for dt in data:

                # Split string to float
                cl, xc, yc, w, h = map(float, dt.split(' '))
                try:
                    cl = int(cl)
                except:
                    print(f'ERROR: could not parse class as integer in file {label_path.name}, exiting.')
                    exit(-1)
                
                if cl >= nc:
                    warnings.append(f'WARNING: found class number {cl} in annotation file {label_path.name} which exceeds or is equal to the number of classes {nc} specified in the YAML file.')

                x = (xc - w / 2)
                y = (yc - h / 2)

                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if x > 1:
                    x = 1
                if y > 1:
                    y = 1

                boxes.append([x, y, w, h, cl])
            
            try:
                dataset.storeDataJsonPair(img_path.name, image_data, { "boxes": boxes, "source": args.source })
            except Exception as e:
                if args.overwrite:
                    try:
                        print('TODO: Check if keys.json is correct')
                        dataset.delete(img_path.name)
                        dataset.storeDataJsonPair(img_path.name, image_data, { "boxes": boxes, "source": args.source })
                    except Exception as e:
                        warnings.append(f'WARNING: Could not overwrite file {img_path.name} in LMDB ({e}), skipping.')
                else:
                    warnings.append(f'WARNING: File {img_path.name} already in LMDB ({e}), skipping.')
            
    print(f"{set_} set done with {len(warnings)} warnings.")
    for w in warnings:
        print(w)

# Generate lmdb yaml
lmdb_yaml_dict = {
    'lmdb': True,
    'nc': nc,
}

for set_ in sets:
    lmdb_yaml_dict[set_] = f'./{set_}'

names = yaml_dict.get('names')
if names:
    lmdb_yaml_dict['names'] = names

with lmdb_path.joinpath('data.yaml').open(mode='w') as lmdb_yaml_file:
    yaml.safe_dump(lmdb_yaml_dict, lmdb_yaml_file)
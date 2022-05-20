import os
import json
import lmdb
import random
import base64
import threading
import cv2 as cv
import numpy as np
from pathlib import Path
from lmdbDataset import LmdbSingleFileDataset, LmdbDataset

class LmdbSingleFileDatasetReadonly:

    def __init__(self, path, map_size=LmdbSingleFileDataset.DefaultMapSize, open_on_init=True):
        path_obj = Path(path)
        if not path_obj.is_dir():
            raise Exception(f'Specified path {path} is not a directory')
        self.path = str(path_obj.absolute())
        del path_obj
        self.map_size = map_size
        self.__keys = None
        self.__lmdb_env = None
        if open_on_init:
            self.open()
    
    def __del__(self):
        self.close()

    def open(self):
        if self.is_open():
            raise Exception('Dataset is already open')
        self.__lmdb_env = lmdb.open(self.path, map_size=self.map_size, readonly=True, lock=False)
        with self.__lmdb_env.begin() as transaction:
            self.__keys = [ key.decode('latin1') for key, _ in transaction.cursor() ]
    
    def close(self):
        if self.is_open():
            self.__lmdb_env.close()
            self.__lmdb_env = None
            self.__keys = None

    def is_open(self):
        return self.__lmdb_env is not None

    def keys(self):
        return self.__keys

    def read_data(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A bytes object containing the read data.
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        if self.__lmdb_env is None: raise Exception("The LMDB environment has not been initialized")
        with self.__lmdb_env.begin() as transaction:
            return transaction.get(key.encode('latin1'))
    
    def read_string(self, key):
        '''
        Reads some data from the dataset as a string.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A string containing the read data.
        '''
        data = self.read_data(key)
        if data is None: return None
        return data.decode('latin1')
    
    def read_json(self, key):
        '''
        Reads some data from the dataset as a JSON object (either a dictionary or an array).
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A JSON containing the read data (either a dictionary or an array).
        '''
        json_string = self.read_string(key)
        if json_string is None: return None
        return json.loads(json_string)
    
    def read_data_json_pair(self, key):
        '''
        Reads some data and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read data (a bytes object).
                2. The read json (either a dictionary or an array).
        '''
        data_json_pair = self.read_json(key)
        if data_json_pair is None: return None, None
        data = base64.decodebytes(data_json_pair["d"].encode('ascii'))
        j = data_json_pair["j"]
        return data, j
    
    def read_image_json_pair(self, key):
        '''
        Reads an image and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read image (a numpy/cv2 array).
                2. The read json (either a dictionary or an array).
        '''
        encoded_image, j = self.read_data_json_pair(key)
        if encoded_image is None: return None, None
        image = cv.imdecode(np.frombuffer(encoded_image, np.uint8), -1)
        return image, j

class LmdbDatasetReadonly:
    
    def __init__(self, path, map_size=LmdbDataset.DefaultMapSize, open_on_init=True):
        self.map_size = map_size
        self.path_obj = Path(path)
        if not self.path_obj.is_dir():
            raise Exception(f'Specified path {path} is not a directory')
        self.__lmdb_datasets = None
        self.__keys_dict = None
        if open_on_init:
            self.open()
    
    def __del__(self):
        self.close()
    
    def open(self):
        if self.is_open():
            raise Exception('Dataset is already open')
        paths = self.find_lmdb_datasets_paths()
        lmdb_datasets = []
        for p in paths:
            lmdb_datasets.append(LmdbSingleFileDatasetReadonly(str(p.absolute()), map_size=self.map_size))
        keys_dict = {}
        for i, d in enumerate(lmdb_datasets):
            for k in d.keys():
                keys_dict[k] = i
        self.__lmdb_datasets = lmdb_datasets
        self.__keys_dict = keys_dict
    
    def close(self):
        for d in self.__lmdb_datasets:
            d.close()
        self.__lmdb_datasets = None
        self.__keys_dict = None
    
    def is_open(self):
        return self.__lmdb_datasets is not None
    
    def find_lmdb_datasets_paths(self):
        lmdb_paths = []
        for i in range(1000000000000):
            name = "lmdb_dataset_{:012d}".format(i)
            path = self.path_obj.joinpath(name)
            if not path.exists() or not path.is_dir():
                break
            lmdb_paths.append(path)
        if not lmdb_paths:
            raise Exception('No LMDB dataset folders found in specified path')
        return lmdb_paths
    
    def get_dataset_containing(self, key):
        i = self.__keys_dict.get(key)
        if i is None: return None
        return self.__lmdb_datasets[i]
    
    def keys(self):
        return self.__keys_dict.keys()
    
    def read_json(self, key):
        '''
        Reads some data from the dataset as a JSON object (either a dictionary or an array).
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A JSON containing the read data (either a dictionary or an array).
        '''
        d = self.get_dataset_containing(key)
        if d is None: return None, None
        return d.read_json(key)
    
    def read_image_json_pair(self, key):
        '''
        Reads an image and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read image (a numpy/cv2 array).
                2. The read json (either a dictionary or an array).
        '''
        d = self.get_dataset_containing(key)
        if d is None: return None, None
        return d.read_image_json_pair(key)

class LmdbMultipleDatasetsReadonly:
    
    def __init__(self, paths, map_size=LmdbDataset.DefaultMapSize, open_on_init=True):
        self.map_size = map_size
        self.path_objs = [Path(p) for p in paths]
        for p in self.path_objs:
            if not p.is_dir():
                raise Exception(f'Specified path {p} is not a directory')
        self.__lmdb_datasets = None
        self.__keys = None
        if open_on_init:
            self.open()
    
    def __del__(self):
        self.close()
    
    def open(self):
        if self.is_open():
            raise Exception('Datasets are already open')
        paths = self.find_lmdb_datasets_paths()
        lmdb_datasets = []
        for p in paths:
            lmdb_datasets.append(LmdbSingleFileDatasetReadonly(str(p.absolute()), map_size=self.map_size))
        keys = []
        for i, d in enumerate(lmdb_datasets):
            for k in d.keys():
                keys.append((i, k))
        self.__lmdb_datasets = lmdb_datasets
        self.__keys = keys
    
    def close(self):
        for d in self.__lmdb_datasets:
            d.close()
        self.__lmdb_datasets = None
        self.__keys = None
    
    def is_open(self):
        return self.__lmdb_datasets is not None
    
    def find_lmdb_datasets_paths(self):
        lmdb_paths = []
        for p in self.path_objs:
            for i in range(1000000000000):
                name = "lmdb_dataset_{:012d}".format(i)
                path = p.joinpath(name)
                if not path.exists() or not path.is_dir():
                    break
                lmdb_paths.append(path)
        if not lmdb_paths:
            raise Exception('No LMDB dataset folders found in specified paths')
        return lmdb_paths
    
    def get_dataset_containing(self, key):
        i = key[0]
        if i >= len(self.__lmdb_datasets):
            return None
        return self.__lmdb_datasets[i]
    
    def keys(self):
        return self.__keys
    
    def read_json(self, key):
        '''
        Reads some data from the dataset as a JSON object (either a dictionary or an array).
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A JSON containing the read data (either a dictionary or an array).
        '''
        d = self.get_dataset_containing(key)
        if d is None: return None, None
        return d.read_json(key[1])
    
    def read_image_json_pair(self, key):
        '''
        Reads an image and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read image (a numpy/cv2 array).
                2. The read json (either a dictionary or an array).
        '''
        d = self.get_dataset_containing(key)
        if d is None: return None, None
        return d.read_image_json_pair(key[1])
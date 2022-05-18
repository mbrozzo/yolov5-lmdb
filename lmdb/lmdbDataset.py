import os
import json
import lmdb
import random
import base64
import threading
import cv2 as cv
import numpy as np


class LmdbSingleFileDataset(object):
    
    # Default map size (100 TB)
    DefaultMapSize = 100* 1024 * 1024 * 1024 * 1024

    def __init__(self, path, mapSize = None, readOnly = False):
        self.__lock = threading.Lock()
        self.__path = path
        self.__mapSize = LmdbSingleFileDataset.DefaultMapSize if mapSize is None else mapSize
        self.__lmdbEnv = None
        self.__readOnly = readOnly
    
    def __enter__(self):
        self.__open()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__close()
    
    def path(self):
        with self.__lock: return self.__path
    
    def __noLock_open(self):
        if not self.__lmdbEnv is None: raise Exception("The LMDB dataset has already been opened")
        self.__lmdbEnv = lmdb.open(self.__path, map_size = self.__mapSize)
    
    def __open(self):
        with self.__lock: self.__noLock_open()
            
    def __noLock_close(self):
        if not self.__lmdbEnv is None:
            self.__lmdbEnv.close()
            self.__lmdbEnv = None
            
    def __close(self):
        with self.__lock: self.__noLock_close()
    
    def __noLock_keys(self):
        if self.__lmdbEnv is None: raise Exception("The LMDB environment has not been initialized")
        with self.__lmdbEnv.begin() as transaction:
            return [ key.decode('latin1') for key, _ in transaction.cursor() ]
    
    def keys(self):
        with self.__lock: return self.__noLock_keys()
    
    def __noLock_storeData(self, key, data):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            data: The data to store (a bytes object).
        '''
        if self.__readOnly: raise Exception("The LMDB dataset has been opened in read-only mode")
        if not type(key) is str: raise Exception("The provided key is not a string")
        if not type(data) is bytes: raise Exception("The provided data is not a bytes object")
        if self.__lmdbEnv is None: raise Exception("The LMDB environment has not been initialized")
        with self.__lmdbEnv.begin(write = True) as transaction:
            transaction.put(key.encode('latin1'), data)
    
    def storeData(self, key, data):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            data: The data to store (a bytes object).
        '''
        with self.__lock: self.__noLock_storeData(key, data)
    
    def __noLock_storeString(self, key, s):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            s: The data to store (a string).
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        if not type(s) is str: raise Exception("The provided data is not a string")
        self.__noLock_storeData(key, s.encode('latin1'))
    
    def storeString(self, key, s):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            s: The data to store (a string).
        '''
        with self.__lock: self.__noLock_storeString(key, s)
    
    def __noLock_storeJson(self, key, j):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            j: The data to store (a json - either a dictionary or an array).
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        self.__noLock_storeString(key, json.dumps(j))
    
    def storeJson(self, key, j):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            j: The data to store (a json - either a dictionary or an array).
        '''
        with self.__lock: self.__noLock_storeJson(key, j)
    
    def __noLock_storeImage(self, key, image, imageFormat = ".jpg"):
        '''
        Stores an image in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            image: The image to store (as a numpy/cv2 array).
            imageFormat: Format with which to encode the image (cv2 string).
        '''
        success, encodedImage = cv.imencode(imageFormat, image)
        if not success: raise Exception("Failed to encode image")
        self.__noLock_storeData(key, encodedImage.tobytes())
    
    def storeImage(self, key, image, imageFormat = ".jpg"):
        '''
        Stores an image in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            image: The image to store (as a numpy/cv2 array).
            imageFormat: Format with which to encode the image (cv2 string).
        '''
        with self.__lock: self.__noLock_storeImage(key, image, imageFormat = imageFormat)
    
    def __noLock_storeDataJsonPair(self, key, data, j):
        '''
        Stores some data and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            data: The data to store (a bytes object).
            j: The json to store (a json - either a dictionary or an array).
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        if not type(data) is bytes: raise Exception("The provided data is not a bytes object")
        dataJsonPair = { "d": base64.encodebytes(data).decode('ascii'), "j": j }
        self.__noLock_storeJson(key, dataJsonPair)
    
    def storeDataJsonPair(self, key, data, j):
        '''
        Stores some data and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            data: The data to store (a bytes object).
            j: The json to store (a json - either a dictionary or an array).
        '''
        with self.__lock: self.__noLock_storeDataJsonPair(key, data, j)
    
    def __noLock_storeStringJsonPair(self, key, s, j):
        '''
        Stores some data and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            s: The string to store (a string).
            j: The json to store (a json - either a dictionary or an array).
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        if not type(s) is str: raise Exception("The provided data is not a string")
        self.__noLock_storeDataJsonPair(key, s.encode('latin1'), j)
    
    def storeStringJsonPair(self, key, s, j):
        '''
        Stores some data and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            data: The string to store (a string).
            j: The json to store (a json - either a dictionary or an array).
        '''
        with self.__lock: self.__noLock_storeStringJsonPair(key, s, j)
    
    def __noLock_storeImageJsonPair(self, key, image, j, imageFormat = ".jpg"):
        '''
        Stores an image and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            image: The image to store (a numpy/cv2 array).
            j: The json to store (a json - either a dictionary or an array).
            imageFormat: Format with which to encode the image (cv2 string).
        '''
        success, encodedImage = cv.imencode(imageFormat, image)
        if not success: raise Exception("Failed to encode image")
        self.__noLock_storeDataJsonPair(key, encodedImage.tobytes(), j)
    
    def storeImageJsonPair(self, key, image, j, imageFormat = ".jpg"):
        '''
        Stores an image and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            image: The image to store (a numpy/cv2 array).
            j: The json to store (a json - either a dictionary or an array).
            imageFormat: Format with which to encode the image (cv2 string).
        '''
        with self.__lock: self.__noLock_storeImageJsonPair(key, image, j, imageFormat = imageFormat)
    
    def __noLock_readData(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A bytes object containing the read data.
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        if self.__lmdbEnv is None: raise Exception("The LMDB environment has not been initialized")
        with self.__lmdbEnv.begin(write = True) as transaction:
            return transaction.get(key.encode('latin1'))
    
    def readData(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A bytes object containing the read data.
        '''
        with self.__lock: return self.__noLock_readData(key)
    
    def __noLock_readString(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A string containing the read data.
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        data = self.__noLock_readData(key)
        if data is None: return None
        return data.decode('latin1')
    
    def readString(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A string containing the read data.
        '''
        with self.__lock: return self.__noLock_readString(key)
    
    def __noLock_readJson(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A JSON containing the read data (either a dictionary or an array).
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        jsonString = self.__noLock_readString(key)
        if jsonString is None: return None
        return json.loads(jsonString)
    
    def readJson(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A JSON containing the read data (either a dictionary or an array).
        '''
        with self.__lock: return self.__noLock_readJson(key)
    
    def __noLock_readImage(self, key):
        '''
        Reads an image from the dataset.
        Parameters:
            key: The key associated with the image to read (a string).
        Returns:
            The loaded image (a numpy/cv2 array).
        '''
        encodedImage = self.__noLock_readData(key)
        if encodedImage is None: return None
        return cv.imdecode(np.frombuffer(encodedImage, np.uint8), -1)
    
    def readImage(self, key):
        '''
        Reads an image from the dataset.
        Parameters:
            key: The key associated with the image to read (a string).
        Returns:
            The loaded image (a numpy/cv2 array).
        '''
        with self.__lock: return self.__noLock_readImage(key)
    
    def __noLock_readDataJsonPair(self, key):
        '''
        Reads some data and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read data (a bytes object).
                2. The read json (either a dictionary or an array).
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        dataJsonPair = self.__noLock_readJson(key)
        if dataJsonPair is None: return None, None
        data = base64.decodebytes(dataJsonPair["d"].encode('ascii'))
        j = dataJsonPair["j"]
        return data, j
    
    def readDataJsonPair(self, key):
        '''
        Reads some data and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read data (a bytes object).
                2. The read json (either a dictionary or an array).
        '''
        with self.__lock: return self.__noLock_readDataJsonPair(key)
    
    def __noLock_readStringJsonPair(self, key):
        '''
        Reads some string and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read string (a string).
                2. The read json (either a dictionary or an array).
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        data, j = self.__noLock_readDataJsonPair(key)
        if data is None: return None, None
        return data.decode('latin1'), j
    
    def readStringJsonPair(self, key):
        '''
        Reads some string and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read string (a string).
                2. The read json (either a dictionary or an array).
        '''
        with self.__lock: return self.__noLock_readStringJsonPair(key)
    
    def __noLock_readImageJsonPair(self, key):
        '''
        Reads an image and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read image (a numpy/cv2 array).
                2. The read json (either a dictionary or an array).
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        encodedImage, j = self.__noLock_readDataJsonPair(key)
        if encodedImage is None: return None, None
        image = cv.imdecode(np.frombuffer(encodedImage, np.uint8), -1)
        return image, j
    
    def readImageJsonPair(self, key):
        '''
        Reads an image and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read image (a numpy/cv2 array).
                2. The read json (either a dictionary or an array).
        '''
        with self.__lock: return self.__noLock_readImageJsonPair(key)
    
    def __noLock_delete(self, key):
        '''
        Deletes an entry from the dataset.
        Parameters:
            key: The key associated with the entry to delete (a string).
        Returns:
            True if the entry was found (and deleted), otherwise False.
        '''
        if not type(key) is str: raise Exception("The provided key is not a string")
        with self.__lmdbEnv.begin(write = True) as transaction:
            return transaction.delete(key.encode('latin1'))
    
    def delete(self, key):
        '''
        Deletes an entry from the dataset.
        Parameters:
            key: The key associated with the entry to delete (a string).
        Returns:
            True if the entry was found (and deleted), otherwise False.
        '''
        with self.__lock: return self.__noLock_delete(key)


class LmdbDataset(object):
    
    # Default map size (1024 MB)
    DefaultMapSize = 1024 * 1024 * 1024
    
    def __init__(self, path, mapSize = None, readOnly = False):
        self.__lock = threading.Lock()
        self.__path = path
        self.__mapSize = LmdbDataset.DefaultMapSize if mapSize is None else mapSize
        self.__readOnly = readOnly
        self.__keys = None
        self.__lmdbDatasets = None
        self.__lastLmdbDatasetIndex = None
    
    def __enter__(self):
        with self.__lock:
            if not os.path.isdir(self.__path): os.mkdir(self.__path)
            self.__noLock_loadKeys()
            self.__noLock_openAll()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        with self.__lock:
            self.__noLock_closeAll(exc_type, exc_value, exc_traceback)
            if not self.__readOnly: self.__noLock_saveKeys()
    
    @staticmethod
    def __ComposeLmdbDatasetName(i):
        return "lmdb_dataset_{:012d}".format(i)
    
    @staticmethod
    def __IsValidLmdbDatasetName(n):
        if not n.startswith("lmdb_dataset_"): return False
        if len(n) != 25: return False
        return True
        
    @staticmethod
    def __ParseLmdbDatasetName(n):
        if not LmdbDataset.__IsValidLmdbDatasetName(n): raise Exception("Bad LMDB Dataset name '{}'".format(n))
        return int(n[13:])
    
    def __noLock_composeLmdbDatasetPath(self, i):
        return os.path.join(self.__path, LmdbDataset.__ComposeLmdbDatasetName(i))
    
    def __composeLmdbDatasetPath(self, i):
        with self.__lock: return self.__noLock_composeLmdbDatasetPath(i)
    
    def __noLock_findLmdbDatasets(self):
        lmdbDatasets = []
        for f in os.listdir(self.__path):
            if LmdbDataset.__IsValidLmdbDatasetName(f):
                lmdbDatasets.append(LmdbDataset.__ParseLmdbDatasetName(f))
        lmdbDatasets.sort()
        return lmdbDatasets
    
    def __findLmdbDatasets(self):
        with self.__lock: self.__noLock_findLmdbDatasets()
    
    def __noLock_lastLmdbDataset(self):
        if self.__lmdbDatasets is None: raise Exception("The LMDB Database has not been opened")
        if len(self.__lmdbDatasets) == 0:
            return None, None
        else:
            if self.__lastLmdbDatasetIndex is None: self.__lastLmdbDatasetIndex = max(self.__lmdbDatasets.keys())
            return self.__lastLmdbDatasetIndex, self.__lmdbDatasets[self.__lastLmdbDatasetIndex]
    
    def __lastLmdbDataset(self):
        with self.__lock: return self.__noLock_lastLmdbDataset()
    
    def __noLock_nextLmdbDatsetIndex(self):
        lastIdx, _ = self.__noLock_lastLmdbDataset()
        if lastIdx is None: return 0
        return lastIdx + 1
    
    def __nextLmdbDatasetIndex(self):
        with self.__lock: return self.__noLock_nextLmdbDatsetIndex()
    
    def __noLock_loadKeys(self):
        keysFilePath = os.path.join(self.__path, "keys.json")
        if os.path.isfile(keysFilePath):
            with open(keysFilePath, 'r') as f:
                self.__keys = json.load(f)
        else:
            self.__keys = {}
        
    def __loadKeys(self):
        with self.__lock: self.__noLock_loadKeys()
    
    def __noLock_saveKeys(self):
        if self.__readOnly: raise Exception("The LMDB dataset has been opened in read-only mode")
        keysFilePath = os.path.join(self.__path, "keys.json")
        with open(keysFilePath, 'w') as f: json.dump(self.__keys, f)
        
    def __saveKeys(self):
        with self.__lock: self.__noLock_saveKeys()
    
    def __noLock_recalculateKeys(self):
        keys = {}
        for i, d in self.__lmdbDatasets.items():
            for k in d.keys():
                keys[k] = i
        self.__keys = keys
        self.__noLock_saveKeys()
    
    def recalculateKeys(self):
        with self.__lock: self.__noLock_recalculateKeys()
    
    def __noLock_keys(self, recalculate = False):
        if recalculate: self.__noLock_recalculateKeys()
        return list(self.__keys.keys())
    
    def keys(self, recalculate = False):
        with self.__lock: return self.__noLock_keys(recalculate)
    
    def iterateKeys(self, random = True, forever = True):
        keys = self.keys()
        while True:
            if random: random.shuffle(keys)
            for k in keys: yield k
            if not forever: break
    
    def __noLock_getDatasetContaining(self, key):
        i = self.__keys.get(key)
        if i is None: return None
        return i, self.__lmdbDatasets[i]
        
    def __noLock_open(self, i):
        lmdbDatasetPath = self.__noLock_composeLmdbDatasetPath(i)
        lmdbDataset = LmdbSingleFileDataset(lmdbDatasetPath, mapSize = self.__mapSize, readOnly = self.__readOnly)
        lmdbDataset.__enter__()
        return lmdbDataset
    
    def __open(self, i):
        with self.__lock: return self.__noLock_open(i)
    
    def __noLock_openAll(self):
        self.__noLock_closeAll()
        self.__lmdbDatasets = { i: self.__noLock_open(i) for i in self.__noLock_findLmdbDatasets() }
        if len(self.__lmdbDatasets) > 0: self.__lastLmdbDatasetIndex = max(self.__lmdbDatasets.keys())
        else: self.__lastLmdbDatasetIndex = None
    
    def __openAll(self):
        with self.__lock: self.__noLock_openAll()
    
    def __noLock_openNext(self):
        nextIdx = self.__noLock_nextLmdbDatsetIndex()
        lmdbDataset = self.__noLock_open(nextIdx)
        self.__lmdbDatasets[nextIdx] = lmdbDataset
        self.__lastLmdbDatasetIndex = nextIdx
        return nextIdx, lmdbDataset
    
    def __openNext(self):
        with self.__lock: return self.__noLock_openNext()
    
    def __noLock_closeAll(self, exc_type = None, exc_value = None, exc_traceback = None):
        if not self.__lmdbDatasets is None:
            for _, d in self.__lmdbDatasets.items():
                d.__exit__(exc_type, exc_value, exc_traceback)
                del d
            self.__lmdbDatasets = None
            self.__lastLmdbDatasetIndex = None
    
    def __closeAll(self, exc_type = None, exc_value = None, exc_traceback = None):
        with self.__lock: self.__noLock_closeAll(exc_type, exc_value, exc_traceback)
    
    def __noLock_getHeadDataset(self):
        if self.__lmdbDatasets is None: raise Exception("The LMDB Dataset has not been opened")
        if self.__lastLmdbDatasetIndex is None:
            _, d = self.__noLock_openNext()
        else:
            d = self.__lmdbDatasets[self.__lastLmdbDatasetIndex]
        return d
    
    def __getHeadDataset(self):
        with self.__lock: return self.__noLock_getHeadDataset()
    
    def __noLock_storeData(self, key, data):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            data: The data to store (a bytes object).
        '''
        try:
            if key in self.__keys: raise Exception("Key already exists")
            self.__noLock_getHeadDataset().storeData(key, data)
        except lmdb.MapFullError:
            _, d = self.__noLock_openNext()
            d.storeData(key, data)
        self.__keys[key] = self.__lastLmdbDatasetIndex
    
    def storeData(self, key, data):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            data: The data to store (a bytes object).
        '''
        with self.__lock: self.__noLock_storeData(key, data)
    
    def __noLock_storeString(self, key, s):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            s: The data to store (a string).
        '''
        try:
            if key in self.__keys: raise Exception("Key already exists")
            self.__noLock_getHeadDataset().storeString(key, s)
        except lmdb.MapFullError:
            _, d = self.__noLock_openNext()
            d.storeString(key, s)
        self.__keys[key] = self.__lastLmdbDatasetIndex
    
    def storeString(self, key, s):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            s: The data to store (a string).
        '''
        with self.__lock: self.__noLock_storeString(key, s)
    
    def __noLock_storeJson(self, key, j):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            j: The data to store (a json - either a dictionary or an array).
        '''
        try:
            if key in self.__keys: raise Exception("Key already exists")
            self.__noLock_getHeadDataset().storeJson(key, j)
        except lmdb.MapFullError:
            _, d = self.__noLock_openNext()
            d.storeJson(key, j)
        self.__keys[key] = self.__lastLmdbDatasetIndex
    
    def storeJson(self, key, j):
        '''
        Stores some data in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            j: The data to store (a json - either a dictionary or an array).
        '''
        with self.__lock: self.__noLock_storeJson(key, j)
    
    def __noLock_storeImage(self, key, image, imageFormat = ".jpg"):
        '''
        Stores an image in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            image: The image to store (as a numpy/cv2 array).
            imageFormat: Format with which to encode the image (cv2 string).
        '''
        try:
            if key in self.__keys: raise Exception("Key already exists")
            self.__noLock_getHeadDataset().storeImage(key, image)
        except lmdb.MapFullError:
            _, d = self.__noLock_openNext()
            d.storeImage(key, image)
        self.__keys[key] = self.__lastLmdbDatasetIndex
    
    def storeImage(self, key, image, imageFormat = ".jpg"):
        '''
        Stores an image in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            image: The image to store (as a numpy/cv2 array).
            imageFormat: Format with which to encode the image (cv2 string).
        '''
        with self.__lock: self.__noLock_storeImage(key, image, imageFormat = imageFormat)
    
    def __noLock_storeDataJsonPair(self, key, data, j):
        '''
        Stores some data and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            data: The data to store (a bytes object).
            j: The json to store (a json - either a dictionary or an array).
        '''
        try:
            if key in self.__keys: raise Exception("Key already exists")
            self.__noLock_getHeadDataset().storeDataJsonPair(key, data, j)
        except lmdb.MapFullError:
            _, d = self.__noLock_openNext()
            d.storeDataJsonPair(key, data, j)
        self.__keys[key] = self.__lastLmdbDatasetIndex
    
    def storeDataJsonPair(self, key, data, j):
        '''
        Stores some data and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            data: The data to store (a bytes object).
            j: The json to store (a json - either a dictionary or an array).
        '''
        with self.__lock: self.__noLock_storeDataJsonPair(key, data, j)
    
    def __noLock_storeStringJsonPair(self, key, s, j):
        '''
        Stores some data and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            s: The string to store (a string).
            j: The json to store (a json - either a dictionary or an array).
        '''
        try:
            if key in self.__keys: raise Exception("Key already exists")
            self.__noLock_getHeadDataset().storeStringJsonPair(key, s, j)
        except lmdb.MapFullError:
            _, d = self.__noLock_openNext()
            d.storeStringJsonPair(key, s, j)
        self.__keys[key] = self.__lastLmdbDatasetIndex
    
    def storeStringJsonPair(self, key, s, j):
        '''
        Stores some data and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            data: The string to store (a string).
            j: The json to store (a json - either a dictionary or an array).
        '''
        with self.__lock: self.__noLock_storeStringJsonPair(key, s, j)
    
    def __noLock_storeImageJsonPair(self, key, image, j, imageFormat = ".jpg"):
        '''
        Stores an image and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            image: The image to store (a numpy/cv2 array).
            j: The json to store (a json - either a dictionary or an array).
            imageFormat: Format with which to encode the image (cv2 string).
        '''
        try:
            if key in self.__keys: raise Exception("Key already exists")
            self.__noLock_getHeadDataset().storeImageJsonPair(key, image, j)
        except lmdb.MapFullError:
            _, d = self.__noLock_openNext()
            d.storeImageJsonPair(key, image, j)
        self.__keys[key] = self.__lastLmdbDatasetIndex
    
    def storeImageJsonPair(self, key, image, j, imageFormat = ".jpg"):
        '''
        Stores an image and a json in the dataset.
        Parameters:
            key: The key to associate with the data (a string).
            image: The image to store (a numpy/cv2 array).
            j: The json to store (a json - either a dictionary or an array).
            imageFormat: Format with which to encode the image (cv2 string).
        '''
        with self.__lock: self.__noLock_storeImageJsonPair(key, image, j, imageFormat = imageFormat)
    
    def __noLock_readData(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A bytes object containing the read data.
        '''
        if self.__lmdbDatasets is None: raise Exception("The LMDB Dataset has not been opened")
        _, d = self.__noLock_getDatasetContaining(key)
        if d is None: return None
        return d.readData(key)
    
    def readData(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A bytes object containing the read data.
        '''
        with self.__lock: return self.__noLock_readData(key)
    
    def __noLock_readString(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A string containing the read data.
        '''
        if self.__lmdbDatasets is None: raise Exception("The LMDB Dataset has not been opened")
        _, d = self.__noLock_getDatasetContaining(key)
        if d is None: return None
        return d.readString(key)
    
    def readString(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A string containing the read data.
        '''
        with self.__lock: return self.__noLock_readString(key)
    
    def __noLock_readJson(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A JSON containing the read data (either a dictionary or an array).
        '''
        if self.__lmdbDatasets is None: raise Exception("The LMDB Dataset has not been opened")
        _, d = self.__noLock_getDatasetContaining(key)
        if d is None: return None
        return d.readJson(key)
    
    def readJson(self, key):
        '''
        Reads some data from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A JSON containing the read data (either a dictionary or an array).
        '''
        with self.__lock: return self.__noLock_readJson(key)
    
    def __noLock_readImage(self, key):
        '''
        Reads an image from the dataset.
        Parameters:
            key: The key associated with the image to read (a string).
        Returns:
            The loaded image (a numpy/cv2 array).
        '''
        if self.__lmdbDatasets is None: raise Exception("The LMDB Dataset has not been opened")
        _, d = self.__noLock_getDatasetContaining(key)
        if d is None: return None
        return d.readImage(key)
    
    def readImage(self, key):
        '''
        Reads an image from the dataset.
        Parameters:
            key: The key associated with the image to read (a string).
        Returns:
            The loaded image (a numpy/cv2 array).
        '''
        with self.__lock: return self.__noLock_readImage(key)
    
    def __noLock_readDataJsonPair(self, key):
        '''
        Reads some data and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read data (a bytes object).
                2. The read json (either a dictionary or an array).
        '''
        if self.__lmdbDatasets is None: raise Exception("The LMDB Dataset has not been opened")
        _, d = self.__noLock_getDatasetContaining(key)
        if d is None: return None, None
        return d.readDataJsonPair(key)
    
    def readDataJsonPair(self, key):
        '''
        Reads some data and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read data (a bytes object).
                2. The read json (either a dictionary or an array).
        '''
        with self.__lock: return self.__noLock_readDataJsonPair(key)
    
    def __noLock_readStringJsonPair(self, key):
        '''
        Reads some string and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read string (a string).
                2. The read json (either a dictionary or an array).
        '''
        if self.__lmdbDatasets is None: raise Exception("The LMDB Dataset has not been opened")
        _, d = self.__noLock_getDatasetContaining(key)
        if d is None: return None, None
        return d.readStringJsonPair(key)
    
    def readStringJsonPair(self, key):
        '''
        Reads some string and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read string (a string).
                2. The read json (either a dictionary or an array).
        '''
        with self.__lock: return self.__noLock_readStringJsonPair(key)
    
    def __noLock_readImageJsonPair(self, key):
        '''
        Reads an image and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read image (a numpy/cv2 array).
                2. The read json (either a dictionary or an array).
        '''
        if self.__lmdbDatasets is None: raise Exception("The LMDB Dataset has not been opened")
        _, d = self.__noLock_getDatasetContaining(key)
        if d is None: return None, None
        return d.readImageJsonPair(key)
    
    def readImageJsonPair(self, key):
        '''
        Reads an image and a json from the dataset.
        Parameters:
            key: The key associated with the data to read (a string).
        Returns:
            A tuple containing:
                1. The read image (a numpy/cv2 array).
                2. The read json (either a dictionary or an array).
        '''
        with self.__lock: return self.__noLock_readImageJsonPair(key)
    
    def __noLock_delete(self, key):
        '''
        Deletes an entry from the dataset.
        Parameters:
            key: The key associated with the entry to delete (a string).
        Returns:
            True if the entry was found (and deleted), otherwise False.
        '''
        if self.__lmdbDatasets is None: raise Exception("The LMDB Dataset has not been opened")
        _, d = self.__noLock_getDatasetContaining(key)
        if d is None: return False
        if not d.delete(key): return False
        self.__keys.pop(key, None)
        return True
    
    def delete(self, key):
        '''
        Deletes an entry from the dataset.
        Parameters:
            key: The key associated with the entry to delete (a string).
        Returns:
            True if the entry was found (and deleted), otherwise False.
        '''
        with self.__lock: return self.__noLock_delete(key)

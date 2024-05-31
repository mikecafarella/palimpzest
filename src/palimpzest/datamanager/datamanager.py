from palimpzest.config import Config
from palimpzest.constants import PZ_DIR
from palimpzest.datasources import DirectorySource, FileSource, MemorySource, UserSource
from palimpzest.elements import UserFunction

import os
import pickle
import sys
import yaml
from threading import Lock

class DataDirectorySingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super(DataDirectorySingletonMeta, cls).__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]    


class CacheService:
    """This class manages the cache for the DataDirectory and other misc PZ components.
       Eventually modify this to be durable and to have expiration policies."""
    def __init__(self):
        self.allCaches = {}

    def getCachedData(self, cacheName, cacheKey):
        return self.allCaches.setdefault(cacheName, {}).get(cacheKey, None)

    def putCachedData(self, cacheName, cacheKey, cacheVal):
        self.allCaches.setdefault(cacheName, {})[cacheKey] = cacheVal

    def rmCachedData(self, cacheName):
        if cacheName in self.allCaches:
            del self.allCaches[cacheName]

# TODO: possibly rename to the PZManager, as it also manages the current config
class DataDirectory(metaclass=DataDirectorySingletonMeta):
    """The DataDirectory is a registry of data sources."""

    def __init__(self):
        self._registry = {}
        self._cache = {}
        self._tempCache = {}
        self.cacheService = CacheService()

        # set up data directory
        self._dir = PZ_DIR
        current_config_path = os.path.join(self._dir, "current_config.yaml")
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
            os.makedirs(self._dir + "/data/registered")
            os.makedirs(self._dir + "/data/cache")
            with open(self._dir + "/data/cache/registry.pkl", "wb") as f:
                pickle.dump(self._registry, f)

            # create default config
            default_config = Config("default")
            default_config.set_current_config()

        # read current config (and dict. of configs) from disk
        self.current_config = None
        if os.path.exists(current_config_path):
            with open(current_config_path, 'r') as f:
                current_config_dict = yaml.safe_load(f)
                self.current_config = Config(current_config_dict['current_config_name'])

        # initialize the file cache directory, defaulting to the system's temporary directory "tmp/pz"
        pz_file_cache_dir = self.current_config.get("filecachedir")
        if not os.path.exists(pz_file_cache_dir):
            os.makedirs(pz_file_cache_dir)

        # Unpickle the registry of data sources
        if os.path.exists(self._dir + "/data/cache/registry.pkl"):
            with open(self._dir + "/data/cache/registry.pkl", "rb") as f:
                self._registry = pickle.load(f)

        # Iterate through all items in the cache directory, and rebuild the table of entries
        for root, _, files in os.walk(self._dir + "/data/cache"):
            for file in files:
                if file.endswith(".cached"):
                    cacheId = file[:-7]
                    self._cache[cacheId] = root + "/" + file

    def getCacheService(self):
        return self.cacheService

    def getConfig(self):
        return self.current_config._load_config()

    def getFileCacheDir(self):
        return self.current_config.get("filecachedir")

    #
    # These methods handle properly registered data files, meant to be kept over the long haul
    #
    def registerLocalDirectory(self, path, dataset_id):
        """Register a local directory as a data source."""
        self._registry[dataset_id] = ("dir", path)
        if not dataset_id.startswith("ephemeral"):
            with open(self._dir + "/data/cache/registry.pkl", "wb") as f:
                pickle.dump(self._registry, f)

    def registerLocalFile(self, path, dataset_id):
        """Register a local file as a data source."""
        self._registry[dataset_id] = ("file", path)
        if not dataset_id.startswith("ephemeral"):
            with open(self._dir + "/data/cache/registry.pkl", "wb") as f:
                pickle.dump(self._registry, f)

    def registerDataset(self, vals, dataset_id):
        """Register an in-memory dataset as a data source"""
        self._registry[dataset_id] = ("memory", vals)
        if not dataset_id.startswith("ephemeral"):
            with open(self._dir + "/data/cache/registry.pkl", "wb") as f:
                pickle.dump(self._registry, f)

    def registerUserSource(self, src:UserSource, dataset_id:str):
        """Register a user source as a data source."""
        self._registry[dataset_id] = ("user", src)
#        # user sources are always ephemeral
        
    def registerUserFunction(self, udf:UserFunction):
        """Register a user function as a data source."""
        self._registry[udf.udfid] = ("udf", udf)
        # user functions are always ephemeral

    def getUserFunction(self, udfid):
        """Return a user function from the registry."""
        if not udfid in self._registry:
            raise Exception("Cannot find user function", udfid, "in the registry.")
        
        entry, rock = self._registry[udfid]
        if entry == "udf":
            return rock
        else:
            raise Exception("Unknown entry type")

    def getRegisteredDataset(self, dataset_id):
        """Return a dataset from the registry."""
        if not dataset_id in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")
        
        entry, rock = self._registry[dataset_id]
        if entry == "dir":
            return DirectorySource(rock, dataset_id)
        elif entry == "file":
            return FileSource(rock, dataset_id)
        elif entry == "memory":
            return MemorySource(rock, dataset_id)
        elif entry == "user":
            src = rock
            return src
        else:
            raise Exception("Unknown entry type")

    def getRegisteredDatasetType(self, dataset_id):
        """Return the type of the given dataset in the registry."""
        if not dataset_id in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")

        entry, _ = self._registry[dataset_id]

        return entry

    def getSize(self, dataset_id):
        """Return the size (in bytes) of a dataset."""
        if not dataset_id in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")
        
        entry, rock = self._registry[dataset_id]
        if entry == "dir":
            # Sum the size in bytes of every file in the directory
            path = rock
            return sum([os.path.getsize(os.path.join(path, name)) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        elif entry == "file":
            # Get the size of the file in bytes
            path = rock
            return os.path.getsize(path)
        elif entry == "memory":
            # get the size of the values in bytes
            return sys.getsizeof(rock)
        elif entry == "user":
            return rock.getSize()
        else:
            raise Exception("Unknown entry type")

    def getCardinality(self, dataset_id):
        """Return the number of records in a dataset."""
        if not dataset_id in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")
        
        entry, rock = self._registry[dataset_id]
        if entry == "dir":
            # Return the number of files in the directory
            path = rock
            return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        elif entry == "file":
            # Return 1
            return 1
        elif entry == "memory":
            # Return the number of elements in the values list
            return len(rock)
        elif entry == "user":
            return rock.getCardinality()
        else:
            raise Exception("Unknown entry type")

    def listRegisteredDatasets(self):
        """Return a list of registered datasets."""
        return self._registry.items()

    def rmRegisteredDataset(self, dataset_id):
        """Remove a dataset from the registry."""
        del self._registry[dataset_id]
        with open(self._dir + "/data/cache/registry.pkl", "wb") as f:
            pickle.dump(self._registry, f)

    #
    # These methods handle cached results. They are meant to be persisted for performance reasons,
    # but can always be recomputed if necessary.
    #
    def getCachedResult(self, cacheId):
        """Return a cached result."""
        if not cacheId in self._cache:
            return None
        with open(self._cache[cacheId], "rb") as f:
            cachedResult = pickle.load(f)
        def iterateOverCachedResult():
            for x in cachedResult:
                yield x
        return iterateOverCachedResult()

    def clearCache(self, keep_registry=False):
        """Clear the cache."""
        self._cache = {}
        self._tempCache = {}

        # Delete all files in the cache directory (except registry.pkl if keep_registry=True)
        for root, dirs, files in os.walk(self._dir + "/data/cache"):
            for file in files:
                if os.path.basename(file) != "registry.pkl" or keep_registry is False:
                    os.remove(root + "/" + file)

    def hasCachedAnswer(self, cacheId):
        """Check if a dataset is in the cache."""
        return cacheId in self._cache

    def openCache(self, cacheId):
        if not cacheId is None and not cacheId in self._cache and not cacheId in self._tempCache:
            self._tempCache[cacheId] = []
            return True
        return False

    def appendCache(self, cacheId, data):
        self._tempCache[cacheId].append(data)

    def closeCache(self, cacheId):
        """Close the cache."""
        filename = self._dir + "/data/cache/" + cacheId + ".cached"
        try:
            with open(filename, "wb") as f:
                pickle.dump(self._tempCache[cacheId], f)
        except pickle.PicklingError:
            print("Warning: Failed to save cache due to pickling error")
            os.remove(filename)
        del self._tempCache[cacheId]
        self._cache[cacheId] = filename

    def exists(self, dataset_id):
        print("Checking if exists", dataset_id, "in", self._registry)
        return dataset_id in self._registry

    def getPath(self, dataset_id):
        if not dataset_id in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")
        entry, path = self._registry[dataset_id]
        return path

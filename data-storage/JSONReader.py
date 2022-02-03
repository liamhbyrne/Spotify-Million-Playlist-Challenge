import json
import os
from concurrent.futures._base import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

from DBManager import DBManager


class JSONReader(DBManager):
    def __init__(self, dir_path : str, n=1000):
        super().__init__()
        self._dir_path = dir_path
        self._n = n  # number of files in directory to parse

    def start(self):
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.parse, "%s/%s" % (self._dir_path, file)) for file in os.listdir(self._dir_path)[:self._n]
                if file.endswith(".json")]
            for future in as_completed(futures):
                future.result()

    def parse(self, file_path : str):
        open_file = open(file_path)
        data = json.load(open_file)
        print(len(data))

if __name__ == '__main__':
    j = JSONReader(r"C:\Users\liamb\Downloads\spotify_million_playlist_dataset\data")
    j.start()
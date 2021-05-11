from src.classes.utils import read_dat_file


class MovieData:
    def __init__(self, path):
        self.data_frame = read_dat_file(path)

    def get_title(self, movieID):
        return self.data_frame[self.data_frame.id == movieID].title.item()

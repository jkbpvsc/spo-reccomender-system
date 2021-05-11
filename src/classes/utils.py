import pandas

def read_dat_file(path):
    return pandas.read_csv(path, sep='\t', encoding='latin-1')

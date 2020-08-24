class DataColumn(object):
    def __init__(self, series, logical_type, tags)
        self.series = series  # pass reference to series, NOT copy
        self.name = series.name
        self.logical_type = logical_type  # pass in string or python Class
        self.dtype = series.dtype  # Physical Type (pandas dtype)
        self.tags = set()

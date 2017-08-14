def get_type(ls):
    if 'H' in ls and not ('b' in ls):
        return 1
    if 'H' in ls:
        return 2
    return 0

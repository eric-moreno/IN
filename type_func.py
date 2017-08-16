def get_type(ls):
    if 'H' in ls and not ('b' in ls):
        return 1
    if 'H' in ls and ls.count('b') == 1:
        return 2
    if 'H' in ls and ls.count('b') == 2:
        return 3
    return 0

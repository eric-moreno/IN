def get_type(ls):
    if 'H' in ls and not ('W+' in ls) and not ('t' in ls) and not ('Z' in ls):
        return 1
    if 'W+' in ls and not ('H' in ls) and not ('t' in ls) and not ('Z' in ls):
        return 2
    if 't' in ls and not ('W+' in ls) and not ('H' in ls) and not ('Z' in ls):
        return 3
    if 'Z' in ls and not ('W+' in ls) and not ('H' in ls) and not ('t' in ls):
        return 4
    return 0

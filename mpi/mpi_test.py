import argparse
import time
import random
parser = argparse.ArgumentParser(description='Get GNN parameters or saved state.')

parser.add_argument('--args',
                    '-a',
                    dest='args',
                    #const=[path + 'np/traininght.npy', path + 'np/targetht.npy', 5, 6, 10, 10, 10, 5, 5, 5],
                    #default=[5, 6, 10, 10, 10, 5, 5, 5],
                    action='store',
                    nargs=2,
                    type=str,
                    help='-a training, target, De, Do, hiddenr1, hiddeno1, hiddenc1, hiddenr2, hiddeno2, hiddenc2')

args = parser.parse_args()
time.sleep(random.randint(1, 10))
out = open("output/" + "-".join(args.args), 'w')
x, y = [float(i) for i in args.args]
n = x ** 2 - y ** 3
out.write(str(n))
out.close()

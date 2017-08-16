from __future__ import print_function
import GraphNet
import argparse
import torch
import ast

parser = argparse.ArgumentParser(description='Get GNN parameters or saved state.')
no_file = "NOFILE"

parser.add_argument('--checkpoint',
                    '-c',
                    dest='checkpoint',
                    const=no_file,
                    default=no_file,
                    action='store',
                    nargs='?',
                    type=str,
                    help='-c /path/to/checkpoint')
parser.add_argument('--args',
                    '-a',
                    dest='args',
                    #const=[5, 6, 10, 10, 10, 5, 5, 5],
                    #default=[5, 6, 10, 10, 10, 5, 5, 5],
                    action='store',
                    nargs=8,
                    type=int,
                    help='-a De Do, hiddenr1, hiddeno1, hiddenc1, hiddenr2, hiddeno2, hiddenc2')
def write_checkpoint(out, training, target, val, val_target, file_name_dict):
    torch.save(training, file_name_dict['training'])
    torch.save(target, file_name_dict['target'])
    torch.save(val, file_name_dict['val'])
    torch.save(val_target, file_name_dict['val_target'])
    args = [gnn.N,
            gnn.n_targets,
            gnn.P,
            gnn.De,
            gnn.Do,
            gnn.hiddenr1,
            gnn.hiddeno1,
            gnn.hiddenc1,
            gnn.hiddenr2,
            gnn.hiddeno2,
            gnn.hiddenc2]
    outf = open(out, 'w')
    outf.write(str(file_name_dict) + '\n')
    outf.write(str(args) + '\n')
    outf.close()
    torch.save(gnn.state_dict(), file_name_dict['gnn'])
    torch.save(optimizer.state_dict(), file_name_dict['optimizer'])

def read_checkpoint(checkpoint):
    inf = open(checkpoint, 'r')
    file_dict, args = [i.strip() for i in inf.readlines()]
    inf.close()
    file_name_dict = ast.literal_eval(file_dict)
    N, n_targets, P, De, Do, hr1, ho1, hc1, hr2, ho2, hc2 = ast.literal_eval(args)
    print(training, target, val, val_target, args, gnn_file)
    gnn = GraphNet.GraphNet(N, n_targets, list(range(P)), De, Do, hr1, ho1, hc1, hr2, ho2, hc2)
    gnn.load_state_dict(torch.load(gnn_file))
    optimizer = optim.Adam(gnn.parameters())
    optimizer.load_state_dict(torch.load(optimizer_file))
    return gnn, optimizer, training, target, val, val_target
    
    
args = parser.parse_args()
checkpoint = args.checkpoint
if checkpoint == no_file:
    print("Not using checkpoint...creating new Graph Network with parameters: %s" % args.args)
else:
    print("Resuming from checkpoint located at %s" % checkpoint)

gnn = GraphNet.GraphNet(10, 4, ['Px'])
print(gnn.state_dict())
write_checkpoint('checkpoint.txt', 'training', 'target', 'val', 'val_target', gnn, 'gnn_file')
read_checkpoint(checkpoint)

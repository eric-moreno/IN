import os, sys, glob

input_file = sys.argv[1]
output_file = input_file.split(".promc")[0] + ".root"
main_path = "/bigdata/shared/HepSIM/"
card = "~/Delphes-3.4.1/cms_notau_nofast.tcl"
print input_file    
l = glob.glob(input_file)
for i in l:
    print "Processing file %s" % i
    out = i.split('.promc')[0] + '.root'
    output_file = out
    if os.path.exists(output_file):
        print "Output file already exists..."
        print "Removing file " + output_file
        os.remove(output_file)
    command = "~/Delphes-3.4.1/./DelphesProMC " + card + " " + output_file + " " + i
    os.system(command)

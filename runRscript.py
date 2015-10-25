
import os.path
import subprocess

rscript = "GRNattr.R"
input_file = "grn.dat"
temp_file = "grn_attr.dat"

# run command:
#   Rscript GRNattr.R grn.dat grn_attr.dat
# create file grn_attr.dat with results
subprocess.check_call(["Rscript", rscript, input_file, temp_file])

# check if output file exists
if not os.path.isfile("grn_attr.dat"):
    raise IOError("Error with " +rscript +" output file "+temp_file)

# parse output of rscipt
try: 
    with open(temp_file, 'r') as f:
        data = f.read()
    data = data.split('\n')
    data = [d.strip().split('\t') for d in data]
except:
    raise IOError("Error with " +rscript +" output file "+temp_file)
# remove temp file
finally:  os.remove(temp_file)

# do something with the results
for d in data[0:-1]: print d



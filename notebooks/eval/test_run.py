import subprocess

script_dir='/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/notebooks/eval/'
script=f'{script_dir}test.py'
process = subprocess.Popen(['python',script], 
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
res=process.communicate()
for line in res[0].decode(encoding='utf-8').split('\n'):
     print(line)

#with process.stdout:
 #   log_subprocess_output(process.stdout)

if process.returncode != 0:
    raise ValueError('Process failed with', process.returncode)

print('Finished wrapper script!')

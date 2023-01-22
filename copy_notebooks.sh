conda activate squidpy

path_code="$WSCL"cross_species_prediction/
path_copy="$WSL"code/cross_species_prediction/notebook_copy/

for f in `find $path_code -name "*ipynb"|grep -Ev 'checkpoints|notebook_copy|temp'`
do
    fn="${f//$path_code/}"  # there used to be '' after slash but I guess this not needed in this system
    fn="${path_copy}${fn}"
    mkdir -p `dirname "$fn"`
    rsync --progress  -u -r -t "$f" "$fn"
    #echo $f 
    #echo $fn
    #echo "*****"
done

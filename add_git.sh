# Check if all ipynb files have matching py file

main_folder=$WSCL/cross_system_integration/
files=`find $main_folder -name *.ipynb`
for file in $files
    do
    # Check only files that re not checkpoints or in the ignore file
    if [[ ! $file =~ "ipynb_checkpoints" ]] && ! $(grep -qFx "$file" "$main_folder"ignore_ipynb.txt) ; then
        file_py="${file/ipynb/py}"
        file_r="${file/ipynb/R}"
        # Find files that do not have py file
        if [ ! -f "$file_py" ] && [ ! -f "$file_r" ]; then
            echo $file
        fi
    fi
done

# Git add in the specified subdirs and specified file types
main_folder=$WSCL/cross_ystem_integration/
cd $main_folder
subdirs=`find . -type d -path "./notebooks*" ! \( -path "*git*" -o -path "*wandb*" -o -path "*.ipynb_checkpoints*" -o -path "*pycache*" -o -path "*egg-info*" \) -print`
echo $subdirs

# If subdirs are ok go on
for folder in . $subdirs
    do
    folder="${folder#./}"
    cd "$main_folder""$folder"
    for format in py sbatch sh txt R yaml md
        do
        unset files
        files=`ls *.$format`
        if [ ! -z "$files" ]
            then
            git add $files
            #echo "$files"
            fi
        done
    done

# We used apptainer based on ml-verse
```
apptainer build --sandbox my_ml_verse docker://rocker/ml-verse
```


# Install python pacjages
```
pip install anndata
```

# Install R packages inside R:

```
install.packages("argparse")
install.packages("Seurat")
install.packages("harmony")
install.packages("anndata")
install.packages("reticulate")
```


# Load required libraries
library(Seurat)
library(argparse)
library(Matrix)
library(anndata)
library(future)

# Use all available CPUs for parallel processing
plan("multicore", workers = availableCores())
options(future.globals.maxSize = 150 * 1024^3)  # 150GB

# Define argument parser
parser <- ArgumentParser()

parser$add_argument('-n', '--name', required=FALSE, type="character", default=NULL,
                    help='name of replicate, if unspecified set to rSEED if seed is given, else to blank string')
parser$add_argument('-s', '--seed', required=FALSE, type="integer", default=NULL,
                    help='random seed, if none it is randomly generated')
parser$add_argument('-po', '--params_opt', required=FALSE, type="character", default='',
                    help='name of optimized params/test purpose')
parser$add_argument('-pa', '--path_adata', required=TRUE, type="character",
                    help='full path to rds object containing Seurat object')
parser$add_argument('-ps', '--path_save', required=TRUE, type="character",
                    help='directory path for saving, creates subdir within it')
parser$add_argument('-sk', '--system_key', required=TRUE, type="character",
                    help='metadata column with system info')
parser$add_argument('-bk', '--batch_key', required=TRUE, type="character",
                    help='metadata column with batch info')
parser$add_argument('-nce', '--n_cells_eval', required=FALSE, type="integer", default=-1,
                    help='Max cells to be used for eval, if -1 use all cells')
parser$add_argument("-rd", "--reduction", help="Seurat reduction", default='cca', type="character")
parser$add_argument('-nl', '--n_latent', required=FALSE, type="integer", default=15,
                    help='number of latent variables')

# Optimization params
parser$add_argument("--k_anchor", required=FALSE, default=5, type="integer", help="Seurat integration k.anchor (default is 5)")
parser$add_argument("--k_weight", required=FALSE, default=100, type="integer", help="Seurat integration k.weight (default is 100)")
parser$add_argument('--harmony_theta', required = FALSE, default = 1.0, type = "numeric", 
                    help = 'Controls the strength of batch effect correction in Harmony (default is 1.0)')


# For testing
parser$add_argument("-t", "--testing", help="Testing mode", default='0', type="character")



# Ignored params
parser$add_argument('-gk', '--group_key', required=TRUE, type="character",
                    help='metadata column with group info')
parser$add_argument('-me', '--max_epochs', required=FALSE, type="integer", default=50,
                    help='max_epochs for training')
parser$add_argument('-edp', '--epochs_detail_plot', required=FALSE, type="integer", default=20,
                    help='Loss subplot from this epoch on')

# args <- parser$parse_args()  # There are excess args
all_args <- parser$parse_known_args()

args <- all_args[[1]]
unknown_args <- all_args[[2]]

print(args)
cat("Unrecognized", unknown_args, "\n")

args$testing <- as.logical(as.integer(args$testing))


##################
# ## For testing in R interactive shell
# args <- list(
#   path_adata = "/om2/user/khrovati/data/cross_system_integration/adipose_sc_sn_updated/adiposeHsSAT_sc_sn.h5ad",
#   path_save = "/home/moinfar/tmp/",
#   testing = FALSE,
#   system_key = "system",
#   batch_key = "sample_id",
#   reduction = "cca",
#   n_cells_eval = 50000,
#   n_latent = 15,
#   harmony_theta = 1.,
#   k_anchor = 5,
#   k_weight=50
# )
##################



# Set name if not provided
if (is.null(args$name)) {
  if (!is.null(args$seed)) {
    args$name <- paste0("r", args$seed)
  } else {
    args$name <- ""
  }
}

# Create saving directory
path_save <- file.path(args$path_save, paste0("seurat_", args$name, "_", 
                                              paste0(sample(c(letters, LETTERS, 0:9), 8, replace=TRUE), collapse=""), 
                                              ifelse(args$testing, "-TEST", "")))
cat("PATH_SAVE=", path_save, "\n")
dir.create(path_save, recursive=TRUE)

# Set seed
if (!is.null(args$seed)) {
  set.seed(args$seed)
}

yaml::write_yaml(args, file.path(path_save, "args.yml"))


# Load data
sdata <- schard::h5ad2seurat(args$path_adata)

# remove _index column present in obs of one data!
if("_index" %in% colnames(sdata@meta.data))
{
  sdata@meta.data$`_index` <- NULL
}


n_hvg <- nrow(sdata)
n_latent = args$n_latent


# If testing, reduce dataset size
if (args$testing) {
  random_idx <- sample(ncol(sdata), 5000)
  sdata <- sdata[, random_idx]

  # remove samples with few cells
  # min 200 as k.filter = 200 in FindIntegrationAnchors
  cell_counts <- table(sdata@meta.data[[args$batch_key]])
  samples_to_keep <- names(cell_counts[cell_counts > 200])
  sdata <- subset(sdata, cells = Cells(sdata)[sdata@meta.data[[args$batch_key]] %in% samples_to_keep])

  cat("Dataset shape after reduction:", nrow(sdata), ncol(sdata), "\n")
}

flush.console()

# Make final batch variable by concating system and batch
sdata@meta.data[["_batch"]] <- paste(sdata@meta.data[[args$system_key]], sdata@meta.data[[args$batch_key]], sep = "_")


### Seurat v5 API ###

all_features = rownames(sdata)
VariableFeatures(sdata) <- all_features

# Splitting
sdata[["RNA"]] <- split(sdata[["RNA"]], f = sdata@meta.data[["_batch"]])

# Pre-processing

#### sdata <- NormalizeData(sdata)  # Our data is always normalized
sdata <- ScaleData(sdata)
sdata <- RunPCA(sdata, npcs = n_latent)


# Integration

if (args$reduction == "cca") {
  min_sample_in_a_batch = min(table(sdata@meta.data[[args$batch_key]]))
  sdata <- IntegrateLayers(
    object = sdata, 
    method = CCAIntegration, 
    features=all_features,
    orig.reduction = "pca", 
    new.reduction = "integrated", 
    k.anchor = args$k_anchor,
    k.weight = args$k_weight,
    verbose = TRUE
  )
}
if (args$reduction == "rpca") {
  min_sample_in_a_batch = min(table(sdata@meta.data[[args$batch_key]]))
  sdata <- IntegrateLayers(
    object = sdata, 
    method = RPCAIntegration, 
    features=all_features,
    orig.reduction = "pca", 
    new.reduction = "integrated", 
    k.anchor = args$k_anchor,
    k.weight = args$k_weight,
    verbose = TRUE
  )
}
if (args$reduction == "harmony") {
  sdata <- IntegrateLayers(
    object = sdata, 
    method = HarmonyIntegration, 
    features=all_features,
    orig.reduction = "pca", 
    new.reduction = "integrated", 
    npcs=n_latent,
    theta = args$harmony_theta,
    verbose = TRUE
  )
}

embed_vactord <- Embeddings(sdata, reduction = "integrated")


# Save seurat object

embed_full_seurat <- CreateSeuratObject(counts = t(embed_vactord), meta.data = sdata@meta.data)
saveRDS(embed_full_seurat, file.path(path_save, "embed_full.rds"))


# Convert to anndata and save
embed_full <- AnnData(X = embed_vactord, obs = sdata@meta.data)

if ((args$n_cells_eval != -1) & (embed_full$n_obs > args$n_cells_eval)) {
  set.seed(0)
  cells_eval <- sample(Cells(sdata), size = args$n_cells_eval)
  embed <- embed_full[embed_full$obs_names %in% cells_eval]$copy()
  write_h5ad(embed_full, file.path(path_save, "embed_full.h5ad"))
} else {
  embed <- embed_full
}

write_h5ad(embed, file.path(path_save, "embed.h5ad"))

cat("Finished integration!")










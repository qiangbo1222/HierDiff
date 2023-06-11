# Coarse-to-Fine: a Hierarchical Diffusion Model for Molecule Generation in 3D

## The instruction for downloading the dataset
We mainly used GEOM drug and crossdock to train and test our model. We provide the processed data for GEOM drug and the raw data for both datasets.

### GEOM Drug


#### Processed dataset
We provide the preprocessed datasets (GEOM) in this [[google drive folder]](https://drive.google.com/file/d/17OQ6PKLZ-J3a5sHdbCqt5PabJ-C9Bg9G/view?usp=sharing)  After downloading the dataset, you should change the specific path in the config files `conf/dataset/denoise.yaml`, `conf/dataset/refine.yaml` and `endiffusion/conf/dataset`.

### Prepare your own GEOM dataset from official dataset (optional)

The offical raw GEOM dataset is avaiable [[here]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF). After downloading the origianl GEOM full dataset and you can prepare your own data via `data_utils/mol_tree.py`. Move the raw data of GEOM drug to `data` and run the following:

```bash
python data_utils/mol_tree.py 'GEOM_drug'
```

### CrossDock
We extract all the bind conformation from Crossdock dataset that are close to the cocrystal structure (RMSD < 1). The sdf file for these conformation can be downloaded from [[google drive folder]](https://drive.google.com/file/d/14vrWKmzXGZ321dgkyzYffbPrlyxkqttH/view?usp=sharing). Move it to `data` and run the following to preprocessed the data to fragment graphs:

```bash
python data_utils/mol_tree.py 'crossdock'
```


### prepare a split index file
Our network that denoise the coarse-grained graph into fine-grained graph requires all input molecules to be encoded as connected fragment graphs. Hence, we generate a train / val /test split index file here to exclude those molecules can be only intepreted as disconnected graph.

```bash
#for GEOM drug
python dataset/split_for_denoise.py --data_dir_base data/GEOM_drugs_trees_blur_correct_adj --save_dir data/geom_denoise_split.pkl
#for crossdock
python dataset/split_for_denoise.py --data_dir_base data/crossdock_blur_trees --save_dir data/crossdock_denoise_split.pkl
```

### CrossDock for Pocket-based generation
We have tested our model for pocket based generation. First, download the crossdock dataset with pocket structures from [[google drive folder]](https://drive.google.com/file/d/10KGuj15mxOJ2FBsduun2Lggzx0yPreEU/view?usp=drive_link). Then, run the following:

```bash
python data_utils/mol_tree.py 'crossdock_cond'
```
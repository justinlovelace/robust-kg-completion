cd data/CN100K/
unzip cn100k.zip
cd ../FB15K-237/
unzip fb15k237.zip
cd ../FB15K-237-Sparse/
unzip fb15k237_sparse.zip
cd ../SNOMED-CT-Core/
unzip snomed_ct_core.zip
cd ../..
python preprocessing/process_datasets.py
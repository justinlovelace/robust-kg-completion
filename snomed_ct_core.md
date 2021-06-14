# SNOMED-CT Core

We provide instructions for recreating the SNOME-CT Core dataset used in our work.

1. Apply for a UMLS license [here](https://www.nlm.nih.gov/databases/umls.html). 
2. Download the 2020AA release of the UMLS. Store the top-level directory (2020AA) in `robust-kg-completion/data/UMLS/`.
3. Download the August 2020 version of the CORE Problem List Subset Data Files from [here](https://www.nlm.nih.gov/healthit/snomedct/archive.html). Store the data files in `robust-kg-completion/data/UMLS/SNOMEDCT_CORE_SUBSET_202008`
4. Run `./scripts/extract_snomed.sh`.


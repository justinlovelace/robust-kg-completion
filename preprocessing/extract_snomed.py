import sys
sys.path.append('..')

import pandas as pd
import os
import typing
import json
from pathlib import Path
from CONSTANTS import SNOMED_CORE_DIR, DATA_DIR, UMLS_SOURCE_DIR

TRAIN_LEN = 502224
VAL_LEN = 71778
TEST_LEN = 143482
KG_LEN = 717488

def extract_cui_relations(cuis: typing.Set[str], df_mrrel: pd.DataFrame) -> pd.DataFrame:
    df_kg = df_mrrel.loc[(df_mrrel['CUI1'].isin(cuis)) &
                         (df_mrrel['CUI2'].isin(cuis))].copy()
    df_kg.drop_duplicates(inplace=True)
    return df_kg


def save_splits(df_kg: pd.DataFrame) -> None:
    data_path = os.path.join(DATA_DIR['SNOMED_CT_CORE'])

    train_idx = pd.read_csv(os.path.join(data_path, 'train_idx.csv'))
    df_train = df_kg.loc[train_idx['idx']]
    assert len(df_train) == TRAIN_LEN
    print('Saving splits...')
    save_path = os.path.join(data_path, 'df_train.csv')
    df_train.to_csv(save_path, index=False)
    print(f'The training data consists of {len(df_train)} facts')


    val_idx = pd.read_csv(os.path.join(data_path, 'val_idx.csv'))
    df_val = df_kg.loc[val_idx['idx']]
    assert len(df_val) == VAL_LEN
    print('Saving splits...')
    save_path = os.path.join(data_path, 'df_valid.csv')
    df_val.to_csv(save_path, index=False)
    print(f'The validation data consists of {len(df_val)} facts')


    test_idx = pd.read_csv(os.path.join(data_path, 'test_idx.csv'))
    df_test = df_kg.loc[test_idx['idx']]
    assert len(df_test) == TEST_LEN
    print('Saving splits...')
    save_path = os.path.join(data_path, 'df_test.csv')
    df_test.to_csv(save_path, index=False)
    print(f'The test data consists of {len(df_test)} facts')



def extract_graph(seed_set: typing.Set[str], df_mrrel: pd.DataFrame) -> None:
    cui_set = seed_set.copy()
    print(f'Seed set using snomed core problems has {len(cui_set)} entities')
    cui_set.update(df_mrrel[df_mrrel['CUI1'].isin(seed_set)]['CUI2'].unique())
    cui_set.update(df_mrrel[df_mrrel['CUI2'].isin(seed_set)]['CUI1'].unique())
    print(len(cui_set))
    df_kg = extract_cui_relations(cui_set, df_mrrel).reset_index(drop=True)
    df_kg = add_idx(df_kg)
    assert len(df_kg) == KG_LEN
    save_path = os.path.join(DATA_DIR['SNOMED_CT_CORE'])
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_path, 'df_kg.csv')
    print(f'Saving graph to {save_path}')
    df_kg.to_csv(save_path, index=False)
    print(df_kg)
    print(f'Saving train/val/test splits...')
    save_splits(df_kg)

def add_idx(df_kg: pd.DataFrame) -> None:
    file_path = os.path.join(DATA_DIR['SNOMED_CT_CORE'], 'entity_idx.json')
    e2idx = json.load(open(file_path))

    file_path = os.path.join(DATA_DIR['SNOMED_CT_CORE'], 'rel_idx.json')
    rel2idx = json.load(open(file_path))


    print('Adding indices...')
    df_kg['CUI1_id'] = df_kg['CUI1'].map(e2idx)
    df_kg['CUI2_id'] = df_kg['CUI2'].map(e2idx)
    df_kg['RELA_id'] = df_kg['RELA'].map(rel2idx).fillna(-1).astype(int)

    return df_kg

def extract_snomed_info():
    df_snomed_core = pd.read_csv(os.path.join(SNOMED_CORE_DIR,'SNOMEDCT_CORE_SUBSET_202008.txt'), delimiter='|')
    df_snomed_core = df_snomed_core[~df_snomed_core.IS_RETIRED_FROM_SUBSET]
    df_snomed_core.dropna(subset=['UMLS_CUI'], inplace=True)
    data_path = os.path.join(UMLS_SOURCE_DIR,'MRREL.RRF')
    df_mrrel = pd.read_csv(data_path, delimiter='|', names=['CUI1', 'CUI2', 'RELA', 'SAB', 'SL', 'SUPPRESS'], usecols=[0, 4, 7, 10, 11, 14])
    # Limit to relations from SNOMEDCT and remove obsolete relations
    df_mrrel.query('SAB == "SNOMEDCT_US" & SL == "SNOMEDCT_US" & SUPPRESS != "O"', inplace=True)
    df_mrrel.drop(columns=['SAB', 'SL', 'SUPPRESS'], inplace=True)
    seed_set = set(df_snomed_core['UMLS_CUI'])
    return df_mrrel, seed_set

def extract_entity_names():
    data_path = os.path.join(UMLS_SOURCE_DIR,'MRCONSO.RRF')
    df_mrconso = pd.read_csv(data_path, sep='|', names=['CUI', 'LAT', 'TS', 'LUI', 'STT', 'ISPREF', 'SAB', 'STR'], usecols=[0, 1, 2, 3, 4, 6, 11, 14])
    df_mrconso.query('LAT == "ENG"', inplace=True)
    unique_entities = json.load(open(os.path.join(DATA_DIR['SNOMED_CT_CORE'], 'entity_idx.json')))
    df_mrconso = df_mrconso[df_mrconso['CUI'].isin(unique_entities)].copy()
    df_mrconso.replace(to_replace='Y', value='A', inplace=True)
    df_mrconso.sort_values(by=['CUI', 'TS', 'STT', 'ISPREF'], inplace=True)
    df_mrconso.drop_duplicates(subset='CUI', keep='first', inplace=True)
    cui2str = {cui:descr for cui, descr in zip(df_mrconso['CUI'], df_mrconso['STR'])}
    file_path = os.path.join(DATA_DIR['SNOMED_CT_CORE'], 'entity_names.json')
    json.dump(cui2str, open(file_path, 'w'))

def main():
    print('Extracting SNOMED relations...')
    df_mrrel, seed_set = extract_snomed_info()
    
    print('Extracting SNOMED graph...')
    extract_graph(seed_set, df_mrrel)

    print('Extracting entity names...')
    extract_entity_names()
    print('done')

if __name__ == '__main__':
    main()
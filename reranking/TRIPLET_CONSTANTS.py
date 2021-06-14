DATA_DIR = {'SNOMED_CT_CORE': '../data/SNOMED-CT-Core/',
            'FB15K_237': '../data/FB15K-237/',
            'FB15K_237_SPARSE': '../data/FB15K-237-Sparse/',
            'CN100K': '../data/CN100K/'}
COLUMN_NAMES = {'SNOMED_CT_CORE': ('CUI2_id', 'RELA_id', 'CUI1_id'),
                'FB15K_237': ('entity1_id', 'rel_id', 'entity2_id'),
                'FB15K_237_SPARSE': ('entity1_id', 'rel_id', 'entity2_id'),
                'CN100K': ('entity1_id', 'rel_id', 'entity2_id')}
BERT_IDS = {'SNOMED_CT_CORE': 'bert_ckpts/SNOMED_CT_CORE',
            'FB15K_237': 'bert_ckpts/FB15K_237',
            'FB15K_237_SPARSE': 'bert_ckpts/FB15K_237',
            'CN100K': 'bert_ckpts/CN100K'}

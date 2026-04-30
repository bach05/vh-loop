# -*- coding: utf-8 -*-
from scripts.data.hf_dataset import canonical_manifest_to_hf_sft

json_file = "/media/iaslab/data_bacchin/panizzolo/paniz_train_04_02_SINGLE.vh_loop.jsonl"

dataset_hf = canonical_manifest_to_hf_sft(json_file)

print(dataset_hf)

print(f'Cahce file: {dataset_hf.cache_files}')
print(f'Fingerprint: {dataset_hf._fingerprint}')

#print first row
sample_0 = dataset_hf[0]
print('Parsing sample 0')
for col in sample_0:
    print(f"{col}: {sample_0[col]}")

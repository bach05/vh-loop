# src/vlmdojo/data/hf_dataset.py

from datasets import Dataset as HFDataset


def canonical_to_hf_sft(samples, adapter):
    rows = []

    for sample in samples:
        rows.append({
            "sample_id": sample.sample_id,
            "messages": adapter.sample_to_messages(sample),
            "images": adapter.sample_to_image_paths(sample),
        })

    return HFDataset.from_list(rows)


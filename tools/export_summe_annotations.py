# -*- coding: utf-8 -*-
"""Export SumMe user summaries into a TSV annotation file for layout parity.

TVSum has an official ydata TSV source. SumMe does not; its official annotations
for this repo are stored in the h5 user_summary arrays. This script exports those
arrays as a repo-local TSV so data/datasets/SumMe mirrors data/datasets/TVSum.
"""
import csv
from pathlib import Path

import h5py
import numpy as np


def main():
    dataset_path = Path('data/datasets/SumMe/eccv16_dataset_summe_google_pool5.h5')
    out_path = Path('data/datasets/SumMe/ydata-anno.tsv')
    meta_path = Path('data/datasets/SumMe/ydata-anno.README.txt')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    videos = 0
    with h5py.File(dataset_path, 'r') as hdf, out_path.open('w', newline='') as out:
        writer = csv.writer(out, delimiter='\t', lineterminator='\n')
        for video_key in sorted(hdf.keys(), key=lambda key: int(key.split('_')[1])):
            user_summary = np.asarray(hdf[f'{video_key}/user_summary'])
            for annotator_idx, summary in enumerate(user_summary):
                score_text = ','.join(str(int(value)) for value in summary.reshape(-1))
                writer.writerow([video_key, f'annotator_{annotator_idx}', score_text])
                rows += 1
            videos += 1

    meta_path.write_text(
        'This file is a repo-local export from '\
        'data/datasets/SumMe/eccv16_dataset_summe_google_pool5.h5:user_summary.\n'
        'Unlike TVSum ydata-anno.tsv, SumMe does not use an official ydata TSV '\
        'in the public PGL-SUM/CA-SUM rank protocol.\n'
        'Rows are: video_key, annotator_id, comma-separated 0/1 frame summary labels.\n',
        encoding='utf-8',
    )
    print(f'wrote {out_path} rows={rows} videos={videos}')


if __name__ == '__main__':
    main()

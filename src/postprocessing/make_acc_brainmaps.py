
import argparse

import nibabel as nib
import numpy as np
import pandas as pd

from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import view_img


def get_arguments():
    """."""
    parser = argparse.ArgumentParser(
        description="Extract validation prediction accuracy metrics per ROI and project into brain volumes."
    )
    parser.add_argument(
        '--metrics_path', required=True, type=str, help='Path to the output metric files ().'
    )
    parser.add_argument(
        '--atlas_path', required=True, type=str, help='Path to the atlas nii.gz file used to extract the timeseries.'
    )
    parser.add_argument(
        '--out_path', required=True, type=str, help="Path where to export the brain maps."
    )
    parser.add_argument(
        '--export_nii', default=False, type=bool, help='if True, export .nii file of R2 scores per ROI for each validation".'
    )

    return parser.parse_args()


def make_brainmaps(args):

    atlas_masker = NiftiLabelsMasker(
        labels_img=args.atlas_path,
        standardize=False,
    )
    atlas_masker.fit()

    results_df = pd.read_csv(
        f"{args.metrics_path}/metrics.csv",
        sep = ",",
    )
    val_df = results_df[results_df['val/brain_loss'].notna()]
    roi_df = val_df[
        sorted([x for x in list(val_df.columns) if 'ROI' in x])
    ]

    for i in range(roi_df.shape[0]):
        nii_file = atlas_masker.inverse_transform(
            roi_df.iloc[i, :].to_numpy()**2,
        )
        if args.export_nii:
            nib.save(
                nii_file,
                f"{args.out_path}_val-{i}.nii.gz",
            )
        vimg = view_img(
            nii_file,
            opacity=0.9,
            cmap="seismic",
            resampling_interpolation='nearest',
            vmax=1.0,
            #symmetric_cmap=False,
        )
        vimg.save_as_html(f"{args.out_path}_val-{i}.html")

        



if __name__ == "__main__":

    args = get_arguments()

    make_brainmaps(args)

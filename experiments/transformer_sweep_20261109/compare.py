"""Compare full validation predictions with historical baseline files."""
import argparse
import csv
import json
from pathlib import Path


def main():
    import awkward as ak
    import numpy as np
    from scipy.stats import wasserstein_distance
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--mse', type=Path, required=True)
    p.add_argument('--emd', type=Path, required=True)
    a = p.parse_args()
    names = ['ttbar_mass','delta_abs_y','cos_K_plus','cos_phi']
    reference = ak.from_parquet(a.mse)
    truth = np.column_stack([ak.to_numpy(reference.targets[n]) for n in names]).astype(float)
    if truth.shape != (260190,4):
        raise ValueError('Historical MSE file must contain the full validation set')
    files = [('historical_MSE',a.mse),('historical_MSE_EMD',a.emd)]
    files += [(f.parent.name,f) for f in sorted(a.results.glob('*/predictions_val.parquet')) if not f.parent.name.startswith('smoke_')]
    rows = []
    for label,path in files:
        data = ak.from_parquet(path)
        target = np.column_stack([ak.to_numpy(data.targets[n]) for n in names]).astype(float)
        if target.shape != truth.shape or not np.allclose(target,truth,atol=0.01,rtol=0):
            raise ValueError(f'Truth/order mismatch: {path}')
        pred = np.column_stack([ak.to_numpy(data.predictions[n]) for n in names]).astype(float)
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError(f'Invalid predictions: {path}')
        status_path = path.parent/'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        for i,n in enumerate(names):
            err = pred[:,i]-truth[:,i]
            rows.append(dict(model=label,observable=n,mae=float(np.abs(err).mean()),rmse=float(np.sqrt((err**2).mean())),bias=float(err.mean()),w1=float(wasserstein_distance(pred[:,i],truth[:,i])),elapsed_seconds=status.get('elapsed_seconds'),completed_full_epochs=status.get('completed_full_epochs'),updates=status.get('updates'),parameters=status.get('parameters')))
    with (a.results/'comparison.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.results/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f'Compared {len(files)} models, {len(truth)} validation events')


if __name__ == '__main__':
    main()

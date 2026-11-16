"""Predict validation events with the audited official VyPER entry point."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import h5py
    import numpy as np
    import torch
    import lightning.pytorch as pl
    from hydra.core.override_parser.overrides_parser import OverridesParser

    root = Path(os.environ.get("RECO_DATA_ROOT", "external")) / 'vyper_full_20261103'
    repo = root / 'source'
    data = root / 'data/aligned_six_global'
    model = root / 'runs/full_seed42/logs/version_2'
    checkpoint = model / 'checkpoints/epoch=97-loss=0.194.ckpt'
    source = data / 'ttbar-2L_val.h5'
    output_dir = root / 'evaluation/epoch97_val_seed42'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == 'c5500033c5eacd0f405b80b6d60eb2988b3dbd4a', commit
    assert not subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only', 'HEAD'], text=True).strip(), 'Official source changed'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert checkpoint.is_file() and (model / 'hparams.yaml').is_file()
    assert list((model / 'checkpoints').glob('epoch=*-loss=*.ckpt')) == [checkpoint], 'Unexpected loss checkpoints; inspect model choice'
    manifest = json.loads((data / 'manifest.json').read_text())
    assert manifest['complete']
    split = manifest['splits']['val']
    assert split['filename'] == source.name
    assert sha256(source) == split['sha256'], 'Validation data hash mismatch'
    rows = split['rows']
    with h5py.File(source) as f:
        assert len(f['INPUTS/GLOBAL']) == rows
    assert source.with_suffix('.db').is_file(), 'Expected existing validation graph cache'
    # Exclusive directory creation protects both completed and partial predictions.
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / 'predictions.h5'
    settings = {
        'datasets.predict_set': str(source),
        'predicting.model_directory': str(model),
        'predicting.model_choice': 'min-loss',
        'predicting.batch_size': 256,
        'predicting.num_sampling_steps': 200,
        'predicting.save_as': str(output),
        'device.num_workers': 2,
        'device.num_devices': 1,
        'device.accelerator': 'gpu',
        'hydra.run.dir': str(output_dir / 'hydra'),
    }
    overrides = [f'{key}={json.dumps(value)}' for key, value in settings.items()]
    OverridesParser.create().parse_overrides(overrides)
    record = dict(status='started', source_commit=commit, seed=42, rows=rows,
                  checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  source=str(source), source_sha256=split['sha256'],
                  script_sha256=sha256(Path(__file__)), settings=settings,
                  job_id=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  sampling='One stochastic prediction per event; 200 diffusion sampling steps')
    report = output_dir / 'prediction_report.json'
    report.write_text(json.dumps(record, indent=2) + '\n')
    print(f'VALIDATION PREDICTION: {rows:,} events; checkpoint epoch 97; batch 256; steps 200', flush=True)
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    pl.seed_everything(42, workers=True)
    sys.argv = ['VyPER.predict', '--config-name=default', *overrides]
    try:
        runpy.run_module('VyPER.predict', run_name='__main__')
        with h5py.File(output) as f:
            g = f['VyPER']
            assert g['Neutrino'].shape == (rows, 3)
            assert g['EdgeIndex'].shape == g['EdgeSoftP'].shape == (rows, 2)
            for start in range(0, rows, 2048):
                nu = g['Neutrino'][start:start+2048]
                edges = g['EdgeIndex'][start:start+2048]
                scores = g['EdgeSoftP'][start:start+2048]
                assert all(len(x) == 2 and np.isfinite(x).all() for x in nu.flat), 'Invalid neutrino prediction'
                for e, s in zip(edges, scores):
                    lengths = [len(x) for x in e] + [len(x) for x in s]
                    assert lengths[0] > 0 and len(set(lengths)) == 1, 'Invalid edge dimensions'
                    assert all(np.isfinite(x).all() for x in s), 'Nonfinite edge score'
        record.update(status='complete', output=str(output), output_sha256=sha256(output))
        print(f'FULL VYPER VALIDATION PREDICTION COMPLETE: {output}', flush=True)
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        report.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()

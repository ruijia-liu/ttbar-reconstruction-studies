"""Freeze the verified Huber+EMD model and cache ALL train/validation events on a Slurm GPU."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

KEYS = ['ttbar_mass', 'delta_abs_y', 'cos_K_plus', 'cos_phi']
INPUTS = ('global', 'jet', 'jet_mask', 'lepton', 'lepton_mask')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def extract_batch(model, batch, device):
    import torch
    import contextlib
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    moved = {k: batch[k].to(device) for k in INPUTS}
    precision = torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' else contextlib.nullcontext()
    with torch.inference_mode(), precision:
        embedding = model.backbone(moved)
        prediction = model.head(embedding)
    return embedding.float().cpu().numpy(), prediction.float().cpu().numpy()


def physical_transform(truth, epsilon=1e-7):
    import numpy as np
    y = np.asarray(truth, dtype=np.float64)
    assert y.ndim == 2 and y.shape[1] == 4 and np.isfinite(y).all()
    assert (y[:, 0] > 0).all() and (np.abs(y[:, 2:]) <= 1+1e-6).all()
    transformed = y.copy()
    transformed[:, 0] = np.log(y[:, 0])
    bounded = np.clip(y[:, 2:], -1+epsilon, 1-epsilon)
    transformed[:, 2:] = np.arctanh(bounded)
    report = dict(atanh_endpoint_adjustments=(bounded != y[:, 2:]).sum(axis=0).tolist(),
                  atanh_max_adjustment=np.max(np.abs(bounded-y[:, 2:]), axis=0).tolist(),
                  original_truth_modified=False)
    assert np.isfinite(transformed).all()
    return transformed, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    here = Path(__file__).resolve().parent
    plan = json.loads((here/'experiment_plan.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output/'status.json'
    status = dict(status='running', full_training_data=True)
    save(status_path, status)
    try:
        sys.path.insert(0, str(args.repo/'src'))
        import awkward as ak
        import numpy as np
        import torch
        from hepsetreg.backbones.backbone import TransformerBackbone
        from hepsetreg.backbones.head import EventRegressor, RegressionHead
        from hepsetreg.datasets.EventDatasetBase import EventDataset, load_arrays
        if not torch.cuda.is_available():
            raise RuntimeError('Feature extraction requires an allocated CUDA GPU')
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('medium')
        for name, expected in plan['source_sha256'].items():
            assert digest(args.repo/'src/hepsetreg'/name) == expected, name
        old = json.loads((args.base/'status.json').read_text())
        checkpoint = Path(old['best_checkpoint'])
        assert digest(checkpoint) == plan['base_artifacts']['checkpoint']['sha256']
        assert digest(args.base/'stats.pt') == plan['base_artifacts']['stats']['sha256']
        assert digest(args.base/'predictions_val.parquet') == plan['base_artifacts']['predictions']['sha256']
        payload = torch.load(args.base/'stats.pt', map_location='cpu', weights_only=False)
        config, stats, features = payload['config'], payload['stats'], payload['features']
        assert features['target'] == KEYS and config['run_id'] == 'huber1_emd01'
        assert config['d_model'] == 128 and config['model'] == 'regression'
        model = EventRegressor(TransformerBackbone(n_global_features=6,n_jet_features=9,n_lepton_features=10,
                                d_model=128,nhead=config['nhead'],num_layers=config['num_layers'],dropout=config['dropout']),
                               RegressionHead(d_model=128,output_dim=4))
        checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict({k[len('model.'):]:v for k,v in checkpoint_data['state_dict'].items() if k.startswith('model.')}, strict=True)
        model.requires_grad_(False).eval().to('cuda')
        target_mean, target_std = [torch.as_tensor(a, dtype=torch.float32) for a in stats['target']]
        canonical_path = here/'canonical_validation.npz'
        assert digest(canonical_path) == plan['canonical_npz_sha256']
        with np.load(canonical_path, allow_pickle=False) as canonical:
            canonical_truth = canonical['targets'].copy()
            assert canonical['keys'].tolist() == KEYS
            assert np.array_equal(canonical['row_index'], np.arange(260190))
        artifacts, checks, split_hashes = {}, {}, {}
        transformed_mean = transformed_std = None
        condition_mean = condition_std = None
        for split, expected_rows in [('train',4692288),('val',260190)]:
            paths = [p for p in [args.data/f'ttbar-2L_{split}_051026_augmented.parquet',args.data/f'ttbar-2L_{split}_four.parquet'] if p.is_file()]
            assert len(paths) == 1
            path = paths[0]
            split_hashes[split] = digest(path)
            assert split_hashes[split] == plan['data_sha256'][split]
            native = ak.from_parquet(path, columns=['targets'])
            truth = np.column_stack([ak.to_numpy(native.targets[k]).astype(np.float64) for k in KEYS])
            del native
            assert truth.shape == (expected_rows,4)
            if split == 'val':
                difference = np.max(np.abs(truth-canonical_truth),axis=0)
                assert np.all(difference <= [.01,5e-4,5e-4,5e-4])
                truth = canonical_truth.copy()
                checks['native_validation_truth_maxabs'] = difference.tolist()
            transformed, adjustment = physical_transform(truth)
            if split == 'train':
                transformed_mean, transformed_std = transformed.mean(axis=0), transformed.std(axis=0)
                assert (transformed_std > 0).all()
            target_u = ((transformed-transformed_mean)/transformed_std).astype(np.float32)
            np.save(args.output/f'{split}_target_u.npy', target_u)
            if split == 'val':
                np.save(args.output/'val_truth.npy',truth)
                np.save(args.output/'row_index_val.npy',np.arange(expected_rows,dtype=np.int64))
            checks[split+'_transform'] = adjustment
            del target_u, transformed, truth
            arrays = load_arrays(str(path),features)
            assert len(arrays[0]) == expected_rows
            counts = ak.to_numpy(ak.num(arrays[1],axis=1))
            assert int(counts.max()) <= payload['max_jets'], 'Would truncate detector input jets'
            dataset = EventDataset(arrays,features=features,stats=stats,max_jets=payload['max_jets'])
            del arrays, counts
            cache = np.lib.format.open_memmap(args.output/f'{split}_condition.npy',mode='w+',dtype=np.float32,shape=(expected_rows,132))
            if split == 'val':
                point = np.lib.format.open_memmap(args.output/'val_point.npy',mode='w+',dtype=np.float32,shape=(expected_rows,4))
            for start in range(0,expected_rows,4096):
                if time.monotonic()-started > 27*60:
                    raise RuntimeError('Cache stage exceeded its internal 27-minute budget')
                stop = min(start+4096,expected_rows)
                batch = dataset.__getitems__(np.arange(start,stop))
                embedding, z = extract_batch(model,batch,'cuda')
                assert np.isfinite(embedding).all() and np.isfinite(z).all()
                cache[start:stop,:128], cache[start:stop,128:] = embedding,z
                if split == 'val':
                    point[start:stop] = (torch.from_numpy(z)*(target_std+1e-8)+target_mean).numpy()
                if start == 0:
                    # Deliberately poisoned targets do not enter the model input whitelist.
                    poisoned = dict(batch, target=torch.full_like(batch['target'],1e20))
                    h2,z2 = extract_batch(model,poisoned,'cuda')
                    assert np.array_equal(h2,embedding) and np.array_equal(z2,z)
                    checks[split+'_target_excluded'] = True
            cache.flush()
            if split == 'train':
                # Fit conditioning preprocessing only on training embeddings; preserve old point_z scales.
                condition_mean = np.asarray(cache[:,:128]).mean(axis=0,dtype=np.float64)
                condition_std = np.asarray(cache[:,:128]).std(axis=0,dtype=np.float64)
                condition_std = np.maximum(condition_std,1e-6)
            for start in range(0,expected_rows,65536):
                stop = min(start+65536,expected_rows)
                cache[start:stop,:128] = ((cache[start:stop,:128]-condition_mean)/condition_std).astype(np.float32)
            cache.flush()
            del cache, dataset
            if split == 'val':
                point.flush()
                historical = ak.from_parquet(args.base/'predictions_val.parquet')
                old_prediction = np.column_stack([ak.to_numpy(historical.predictions[k]) for k in KEYS])
                assert np.array_equal(ak.to_numpy(historical.row_index),np.arange(expected_rows))
                old_truth = np.column_stack([ak.to_numpy(historical.targets[k]) for k in KEYS])
                assert np.array_equal(old_truth,canonical_truth)
                difference = np.max(np.abs(point.astype(float)-old_prediction),axis=0)
                assert np.all(difference <= [0.002,1e-5,1e-5,1e-5]), ('Frozen baseline changed',difference)
                checks['point_prediction_maxabs'] = difference.tolist()
                checks['point_prediction_exact'] = bool(np.array_equal(point,old_prediction))
                # The comparison uses the original saved baseline, not an altered prediction set.
                point[:] = old_prediction.astype(np.float32)
                point.flush()
                del point, historical, old_prediction, old_truth
            gc.collect()
            torch.cuda.empty_cache()
            print('Cached full',split,expected_rows,'events; seconds',time.monotonic()-started,flush=True)
        transform_record = dict(keys=KEYS,mean=transformed_mean.tolist(),std=transformed_std.tolist(),
                                transform=['log','identity','atanh','atanh'],atanh_epsilon=1e-7,
                                fitted_on='full training truth only',physical_prediction_clipping=False,
                                point_z_mean=target_mean.tolist(),point_z_std=target_std.tolist(),
                                condition_embedding_mean=condition_mean.tolist(),condition_embedding_std=condition_std.tolist(),
                                condition_last_four='original standardized frozen point prediction, used only as conditioning; anchor_last=False')
        save(args.output/'transforms.json',transform_record)
        for path in sorted(args.output.glob('*.npy')):
            arr = np.load(path,mmap_mode='r',allow_pickle=False)
            assert np.isfinite(arr).all()
            artifacts[path.name] = dict(sha256=digest(path),shape=list(arr.shape),dtype=str(arr.dtype),bytes=path.stat().st_size)
            del arr
        artifacts['transforms.json'] = dict(sha256=digest(args.output/'transforms.json'),bytes=(args.output/'transforms.json').stat().st_size)
        elapsed = time.monotonic()-started
        manifest = dict(status='complete',train_rows=4692288,val_rows=260190,artifacts=artifacts,
                        data_sha256=split_hashes,canonical_truth_sha256=plan['canonical_npz_sha256'],
                        frozen_checkpoint=dict(path=str(checkpoint),sha256=digest(checkpoint)),
                        point_alignment_verified=True,source_prediction_sha256=plan['base_artifacts']['predictions']['sha256'],
                        checks=checks,encoder_frozen=True,encoder_eval=True,encoder_precision='bf16 autocast',
                        source_sha256=plan['source_sha256'],elapsed_seconds=elapsed,gpu=torch.cuda.get_device_name(),
                        training_events_subsampled=False,validation_fitted_preprocessing=False,
                        train_row_order='source Parquet order; implicit arange(train_rows)',
                        validation_row_order='source Parquet order; explicit row_index_val.npy')
        save(args.output/'manifest.json',manifest)
        status.update(status='complete',elapsed_seconds=elapsed,manifest_sha256=digest(args.output/'manifest.json'))
        save(status_path,status)
    except Exception:
        status.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        save(status_path,status)
        raise


if __name__ == '__main__':
    main()

"""Train-only top/antitop supervision and differentiable physical consistency.

Eight auxiliary coordinates are px,py,pz,log(m/GeV) for top then antitop.
Reconstruction predicts a positive mass and derives E, so every reconstructed
parent is future timelike. No fixed top mass or neutrino truth is supplied to
the forward model. Source targets are only supervised labels, never inputs.
"""
from __future__ import annotations

import numpy as np
import torch

MAIN_KEYS = ['ttbar_mass', 'delta_abs_y', 'cos_K_plus', 'cos_phi']
P4_KEYS = ['top_px', 'top_py', 'top_pz', 'top_E',
           'antitop_px', 'antitop_py', 'antitop_pz', 'antitop_E']
AUX_KEYS = ['top_px', 'top_py', 'top_pz', 'top_log_mass',
            'antitop_px', 'antitop_py', 'antitop_pz', 'antitop_log_mass']
STD_FLOORS = np.array([1., 1., 1., .05, 1., 1., 1., .05], dtype=np.float64)


def p4_to_auxiliary(p4):
    """Validate all source rows; never remove or clip nonphysical truth."""
    x = np.asarray(p4, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 8 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty [N,8] source four-vectors')
    result = x.copy()
    mass_ranges = []
    for start in (0, 4):
        energy = x[:, start+3]
        mass2 = energy*energy - np.sum(x[:, start:start+3]**2, axis=1)
        invalid = (energy <= 0) | (mass2 <= 0) | ~np.isfinite(mass2)
        if invalid.any():
            rows = np.flatnonzero(invalid)[:8].tolist()
            raise ValueError(f'Nonfuture/non-timelike source parent at chunk rows {rows}; no rows dropped')
        mass = np.sqrt(mass2)
        result[:, start+3] = np.log(mass)
        mass_ranges.append([float(mass.min()), float(mass.max())])
    values = result.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Auxiliary coordinates overflow float32')
    return values, mass_ranges


def load_auxiliary(path, batch_size=262144):
    """Read only the eight parent labels plus four aligned final labels.

    This accepts the augmented train/validation file, which contains delta_abs_y.
    Counts/order match the Parquet row order; the caller verifies file hashes and
    main-label alignment against its already loaded data tensors.
    """
    import awkward as ak
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    fields = {field.name for field in parquet.schema_arrow.field('targets').type}
    missing = set(P4_KEYS+MAIN_KEYS) - fields
    if missing:
        raise ValueError('Missing supervised target fields: '+str(sorted(missing)))
    rows = parquet.metadata.num_rows
    auxiliary = np.empty((rows, 8), dtype=np.float32)
    targets = np.empty((rows, 4), dtype=np.float64)
    lower = np.full(2, np.inf); upper = np.full(2, -np.inf)
    maximum_main_difference = np.zeros(2,dtype=np.float64)
    maximum_tolerance_ratio = np.zeros(2,dtype=np.float64)
    offset = 0
    for batch in parquet.iter_batches(batch_size=batch_size,
                                     columns=['targets.'+key for key in P4_KEYS+MAIN_KEYS]):
        record = ak.from_arrow(batch).targets
        p4 = np.column_stack([ak.to_numpy(record[key]) for key in P4_KEYS])
        values, mass_ranges = p4_to_auxiliary(p4)
        truth = np.column_stack([ak.to_numpy(record[key]) for key in MAIN_KEYS]).astype(np.float64)
        if not np.isfinite(truth).all():
            raise ValueError('Nonfinite primary target in auxiliary source')
        # Independently check units and the physical meaning of both shared
        # observables before adding a consistency loss to the model.
        parents = np.asarray(p4,dtype=np.float64).reshape(-1,2,4)
        total = parents.sum(axis=1)
        pair_mass2 = total[:,3]**2-np.sum(total[:,:3]**2,axis=1)
        transverse_mass2 = parents[:,:,3]**2-parents[:,:,2]**2
        if np.any(pair_mass2<=0) or np.any(transverse_mass2<=0):
            raise ValueError('Source pair or transverse mass is nonphysical')
        y = np.arcsinh(parents[:,:,2]/np.sqrt(transverse_mass2))
        expected = np.column_stack([np.sqrt(pair_mass2),np.abs(y[:,0])-np.abs(y[:,1])])
        differences = np.abs(expected-truth[:,:2])
        difference = np.max(differences,axis=0)
        maximum_main_difference = np.maximum(maximum_main_difference,difference)
        tolerance = np.column_stack([.02+1e-4*np.abs(truth[:,0]),np.full(len(truth),5e-4)])
        ratio = np.max(differences/tolerance,axis=0)
        maximum_tolerance_ratio = np.maximum(maximum_tolerance_ratio,ratio)
        if np.any(ratio>1):
            raise ValueError('Source four-vectors disagree with primary mass/rapidity labels: '+str(difference))
        stop = offset+len(values)
        auxiliary[offset:stop] = values; targets[offset:stop] = truth
        ranges = np.asarray(mass_ranges)
        lower = np.minimum(lower, ranges[:, 0]); upper = np.maximum(upper, ranges[:, 1])
        offset = stop
    if offset != rows or rows == 0:
        raise ValueError('Incomplete/empty auxiliary source')
    return dict(auxiliary=auxiliary, targets=targets,
                report=dict(rows=rows,source_fields=P4_KEYS,auxiliary_keys=AUX_KEYS,
                    mass_ranges_gev=np.column_stack([lower,upper]).tolist(),
                    main_p4_consistency_maxabs=maximum_main_difference.tolist(),
                    main_p4_consistency_max_tolerance_ratio=maximum_tolerance_ratio.tolist(),
                    main_p4_consistency_tolerance=dict(mass_atol_gev=.02,mass_rtol=1e-4,delta_atol=5e-4,
                        reason='Source four-vectors and stored labels may be rounded separately in float32, especially at large boosts'),
                    nonphysical_rows=0,rows_removed=0,source_truth_clipping=False))


def fit_aux_stats(auxiliary):
    """Fit on the full training split only; validation is never passed here."""
    x = np.asarray(auxiliary)
    if x.ndim != 2 or x.shape[1] != 8 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty [N,8] training auxiliary coordinates')
    mean = x.mean(axis=0, dtype=np.float64)
    raw_std = x.std(axis=0, dtype=np.float64)
    std = np.maximum(raw_std, STD_FLOORS)
    return dict(mean=mean.tolist(),std=std.tolist(),raw_std=raw_std.tolist(),
                std_floors=STD_FLOORS.tolist(),fit_rows=len(x),fit_split='train',keys=AUX_KEYS)


def standardize_aux(auxiliary, stats):
    x = np.asarray(auxiliary)
    mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
    if x.ndim != 2 or x.shape[1] != 8 or mean.shape != (8,) or std.shape != (8,):
        raise ValueError('Wrong auxiliary/statistics shape')
    if not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid auxiliary coordinates/statistics')
    z = ((x-mean)/std).astype(np.float32)
    if not np.isfinite(z).all():
        raise ValueError('Standardized auxiliary coordinates are not finite')
    return z


def reconstruct_observables(aux_z, aux_mean, aux_std, main_mean, main_effective_std):
    """Return differentiable standardized [m_ttbar, |y_top|-|y_antitop|].

    Float64 physics avoids cancellation for boosted parents; the final values
    are float32 for the shared loss. exp(logmass) is never clipped. A 1e-12
    floor on the pair mass squared only guards roundoff of an analytically
    timelike sum; invalid/nonfinite output raises instead of hiding events.
    Rapidity asinh(pz/mT) avoids subtracting nearly equal E and pz.
    """
    if aux_z.ndim != 2 or aux_z.shape[1] != 8 or not len(aux_z):
        raise ValueError('Expected nonempty [N,8] predicted auxiliary coordinates')
    device = aux_z.device
    with torch.autocast(device_type=device.type, enabled=False):
        am = torch.as_tensor(aux_mean, device=device, dtype=torch.float64)
        ast = torch.as_tensor(aux_std, device=device, dtype=torch.float64)
        mm = torch.as_tensor(main_mean, device=device, dtype=torch.float64)
        mst = torch.as_tensor(main_effective_std, device=device, dtype=torch.float64)
        if am.shape != (8,) or ast.shape != (8,) or mm.shape != (4,) or mst.shape != (4,):
            raise ValueError('Wrong reconstruction statistic dimensions')
        if not all(bool(torch.isfinite(s).all()) for s in (am,ast,mm,mst)):
            raise ValueError('Reconstruction statistics must be finite')
        if not bool((ast > 0).all()) or not bool((mst > 0).all()):
            raise ValueError('Reconstruction scales must be positive')
        raw = aux_z.double()*ast+am
        parents = raw.reshape(-1, 2, 4)
        momenta = parents[:, :, :3]
        masses2 = torch.exp(2*parents[:, :, 3])
        if not bool(torch.isfinite(masses2).all()) or not bool((masses2>0).all()):
            raise FloatingPointError('Predicted positive parent masses underflowed/overflowed; no clipping applied')
        energies = torch.sqrt(momenta.square().sum(dim=2)+masses2)
        total_p = momenta.sum(dim=1)
        pair_mass2 = energies.sum(dim=1).square()-total_p.square().sum(dim=1)
        pair_mass = torch.sqrt(pair_mass2.clamp_min(1e-12))
        transverse_mass = torch.sqrt(momenta[:, :, :2].square().sum(dim=2)+masses2)
        rapidity = torch.asinh(momenta[:, :, 2]/transverse_mass)
        delta = rapidity[:, 0].abs()-rapidity[:, 1].abs()
        physical = torch.stack([pair_mass, delta], dim=1)
        result = ((physical-mm[:2])/mst[:2]).float()
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('Physical auxiliary reconstruction is nonfinite; predictions not clipped')
        return result

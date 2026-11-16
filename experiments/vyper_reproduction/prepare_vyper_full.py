"""Verify complete paired inputs, then project GLOBAL for official VyPER.
Run with the existing HepSetReg Python (h5py, awkward, pyarrow, yaml installed).
No event selection, model training, or test-set model scoring occurs here.
"""
import os
import hashlib
import json
from pathlib import Path
import subprocess

import awkward as ak
import h5py
import numpy as np
import pyarrow.parquet as pq
import yaml

FIELDS = ['njet', 'nbTagged', 'nelectron', 'nmuon', 'met_met', 'met_phi']
GLOBALS = ['global_njet', 'global_nbtagged', 'global_nelectron', 'global_nmuon', 'global_met_met', 'global_met_phi']
HASHES = {
 'train': 'eb148487e63f4719e510a1d6b57402a654f03bf60efd4653db5ec9aa85d337d4',
 'val': '0af12b08e57eaeb181d3a6ab20b6532ac79212cd90ecc3b2f7d4416783b7d563',
 'test': 'fafeddf9e9cf1d5edb09ff3dff7dbf010506b8743b1c40894df43635f55975f1',
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def require(ok, message):
    if not ok: raise RuntimeError(message)

def verify_alignment(h5path, parquetpath):
    offset = 0
    with h5py.File(h5path, 'r') as source:
        pf = pq.ParquetFile(parquetpath)
        count = len(source['INPUTS/GLOBAL'])
        require(count == pf.metadata.num_rows, 'Event counts differ')
        for batch in pf.iter_batches(batch_size=20000, columns=['inputs']):
            inp = ak.from_arrow(batch)['inputs']
            stop = offset + len(inp)
            for obj, prefix in [('JET','jet'),('ELECTRON','el'),('MUON','mu')]:
                data = source['INPUTS/'+obj][offset:stop]
                mask = np.isfinite(data['pt'])
                require(np.array_equal(mask, np.arange(mask.shape[1])[None,:] < mask.sum(1)[:,None]), f'{obj}: non-trailing padding')
                columns = ['e','eta','phi','pt','btag' if obj == 'JET' else 'charge']
                for name in columns:
                    col = prefix + '_' + ('bTag' if name == 'btag' else name)
                    require(np.array_equal(mask.sum(1), ak.to_numpy(ak.num(inp[col]))), f'Object count/order mismatch: {col}, chunk {offset}')
                    require(np.array_equal(np.isfinite(data[name]), mask), f'Padding/nonfinite mismatch: {obj}/{name}')
                    values = ak.to_numpy(ak.flatten(inp[col]), allow_missing=False)
                    require(np.array_equal(data[name][mask], values), f'Input values/order mismatch: {col}, chunk {offset}; no rows removed')
            glob = source['INPUTS/GLOBAL'][offset:stop]
            for name, col in zip(FIELDS, GLOBALS):
                require(np.array_equal(glob[name].reshape(-1), ak.to_numpy(inp[col], allow_missing=False)), f'Global mismatch: {col}, chunk {offset}')
            offset = stop
            if offset % 200000 == 0 or offset == count:
                print(f'ALIGNMENT {h5path.name}: {offset:,}/{count:,}', flush=True)
        require(offset == count, 'Incomplete alignment scan')
    return count

def equal_array(left, right):
    if left.dtype.names:
        return all(equal_array(left[n], right[n]) for n in left.dtype.names)
    return np.array_equal(left, right, equal_nan=True) if left.dtype.kind in 'fc' else np.array_equal(left,right)

def project_and_verify(source_path, destination):
    require(not destination.exists(), f'Output exists: {destination}; do not overwrite it')
    temporary = destination.with_suffix('.h5.writing')
    with h5py.File(source_path,'r') as src, h5py.File(temporary,'w') as out:
        for key, value in src.attrs.items(): out.attrs[key] = value
        for key in src:
            if key != 'INPUTS': src.copy(key,out)
        inputs = out.create_group('INPUTS')
        for key,value in src['INPUTS'].attrs.items(): inputs.attrs[key]=value
        for key in src['INPUTS']:
            if key != 'GLOBAL': src.copy('INPUTS/'+key,inputs,name=key)
        original = src['INPUTS/GLOBAL']
        dtype = np.dtype([(name,original.dtype.fields[name][0]) for name in FIELDS])
        target = inputs.create_dataset('GLOBAL',shape=original.shape,dtype=dtype)
        for key,value in original.attrs.items(): target.attrs[key]=value
        for start in range(0,len(original),20000):
            block=original[start:start+20000]
            projected=np.empty(block.shape,dtype=dtype)
            for name in FIELDS: projected[name]=block[name]
            target[start:start+len(block)]=projected
    # Verify all datasets, including assignment/neutrino labels and metadata.
    with h5py.File(source_path,'r') as src, h5py.File(temporary,'r') as out:
        names=[]
        src.visititems(lambda name,obj: names.append(name) if isinstance(obj,h5py.Dataset) else None)
        other=[]
        out.visititems(lambda name,obj: other.append(name) if isinstance(obj,h5py.Dataset) else None)
        require(names==other,'Dataset paths changed')
        for name in names:
            a,b=src[name],out[name]
            require(a.shape==b.shape,f'Shape changed: {name}')
            if name=='INPUTS/GLOBAL': require(list(b.dtype.names)==FIELDS,'Wrong projected global schema')
            else: require(a.dtype==b.dtype,f'Dtype changed: {name}')
            if not a.shape:
                require(equal_array(np.asarray(a[()]),np.asarray(b[()])),f'Scalar changed: {name}')
                continue
            for start in range(0,len(a),20000):
                x,y=a[start:start+20000],b[start:start+20000]
                ok=all(equal_array(x[f],y[f]) for f in FIELDS) if name=='INPUTS/GLOBAL' else equal_array(x,y)
                require(ok,f'Copied values changed: {name}, row {start}')
    temporary.replace(destination)

def main():
    root=Path(os.environ.get("RECO_DATA_ROOT", "external")) / 'vyper_full_20261103'
    teacher=Path(os.environ.get("RECO_DATA_ROOT", "external")) / 'hepsetreg_benchmark_20261027/teacher_051026_snapshot/derived'
    manifest_path=teacher/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    require(manifest.get('complete') is True,'Teacher data manifest incomplete')
    repo=root/'source'
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    require(commit=='c5500033c5eacd0f405b80b6d60eb2988b3dbd4a','Unexpected official source commit')
    # Save the physical config hash; schema uses the audited release's six fields.
    yaml.safe_load((repo/'configs/default.yaml').read_text())
    dest=root/'data/aligned_six_global'
    dest.mkdir(parents=True,exist_ok=True)
    report={'code_commit':commit,'config_sha256':digest(repo/'configs/default.yaml'),
            'teacher_manifest_sha256':digest(manifest_path),'script_sha256':digest(__file__),
            'comparison':'All events, exact equality of 21 common detector input fields, original object and event order',
            'limitation':'Input alignment does not establish absence of duplicate events across splits or equivalence of all truth conventions.',
            'splits':{}}
    for split in ('val','test','train'):
        raw=root/'data/raw'/f'ttbar-2L_{split}.h5'
        record=manifest['splits'][split]
        parquet=teacher/record['filename']
        print(f'Checking source hashes: {split}',flush=True)
        require(digest(raw)==HASHES[split],f'Official {split} SHA256 mismatch')
        require(digest(parquet)==record['sha256'],f'Teacher {split} SHA256 mismatch')
        count=verify_alignment(raw,parquet)
        output=dest/f'ttbar-2L_{split}.h5'
        receipt=dest/f'{split}.json'
        if output.exists():
            old=json.loads(receipt.read_text())
            require(old['source_sha256']==HASHES[split] and digest(output)==old['sha256'],'Existing derived file verification failed')
        else:
            project_and_verify(raw,output)
            old={'rows':count,'filename':output.name,'sha256':digest(output),'source_sha256':HASHES[split], 'all_original_non_GLOBAL_datasets_preserved':True}
            receipt.write_text(json.dumps(old,indent=2)+'\n')
        report['splits'][split]=old
        print(f'ALIGNED AND PREPARED: {split}, {count:,} events',flush=True)
    report['complete']=True
    (dest/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'VYPER DATA READY: {dest}',flush=True)

if __name__=='__main__': main()

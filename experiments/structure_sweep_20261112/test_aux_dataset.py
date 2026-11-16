import unittest
import numpy as np
import torch
from aux_dataset import AuxiliaryDataset

class DummyDataset:
    def __len__(self):return 17
    def __getitem__(self,i):return {'global':torch.tensor([i]),'target':torch.full((4,),float(i))}
    def __getitems__(self,idx):
        idx=torch.as_tensor(idx)
        return {'global':idx[:,None],'target':idx[:,None].expand(-1,4).float()}

class AuxiliaryDatasetTests(unittest.TestCase):
    def test_shuffled_repeated_indices_keep_labels_aligned(self):
        aux=np.repeat(np.arange(17)[:,None],8,axis=1)
        ds=AuxiliaryDataset(DummyDataset(),aux,np.arange(17)%2)
        idx=np.array([16,2,8,2,0])
        b=ds.__getitems__(idx)
        torch.testing.assert_close(b['global'][:,0].float(),b['aux_target'][:,0])
        torch.testing.assert_close(b['tail_label'],torch.tensor(idx%2,dtype=torch.float32))
        self.assertEqual(ds[5]['aux_target'][0],5)
    def test_dataloader_batch_contract(self):
        aux=np.repeat(np.arange(17)[:,None],8,axis=1)
        ds=AuxiliaryDataset(DummyDataset(),aux,np.arange(17)%2)
        loader=torch.utils.data.DataLoader(ds,batch_size=6,shuffle=True,collate_fn=lambda x:x)
        ids=[]
        for b in loader:
            self.assertIsInstance(b,dict)
            torch.testing.assert_close(b['target'][:,0],b['aux_target'][:,0])
            ids.extend(b['global'][:,0].tolist())
        self.assertEqual(sorted(ids),list(range(17)))
    def test_wrong_rows_rejected(self):
        with self.assertRaises(ValueError):AuxiliaryDataset(DummyDataset(),np.zeros((16,8)),np.zeros(17))

if __name__=='__main__':unittest.main()

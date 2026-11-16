import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from common import Correction,predict,checkpoint,point_shape
from evaluation import write_predictions

class CommonTests(unittest.TestCase):
    def test_zero_is_exact_physical_baseline_even_with_distinct_cached_anchor(self):
        c=torch.randn(31,132);base=np.random.default_rng(1).normal(size=(31,4)).astype(np.float32)
        actual=predict(Correction(),c,base,[150,.6,.5,.5],batch_size=7)
        np.testing.assert_array_equal(actual,base.astype(np.float64))
    def test_bounded_correction_and_checkpoint(self):
        model=Correction();x=torch.randn(20,132)
        with torch.no_grad():model.output.weight.fill_(.3)
        pred,delta,_=model(x)
        self.assertTrue(bool((delta.abs()<=.5).all()))
        torch.testing.assert_close(pred,x[:,-4:]+delta)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt';checkpoint(path,model,dict(epoch=0))
            restored=Correction();restored.load_state_dict(torch.load(path,weights_only=False)['state_dict'])
            torch.testing.assert_close(restored(x)[0],pred,rtol=0,atol=0)
    def test_training_update_moves_zero_initialized_output(self):
        model=Correction();x=torch.randn(32,132);target=x[:,-4:]+.1
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        p,_,_=model(x);point,shape=point_shape(p,target);(point+.1*shape).backward();opt.step()
        self.assertGreater(float(model(x)[1].abs().sum().detach()),0)
    def test_noncontiguous_physical_export(self):
        a=np.arange(80,dtype=np.float64).reshape(20,4)[::-1]
        with tempfile.TemporaryDirectory() as d:write_predictions(Path(d)/'x.parquet',a+.1,a)

if __name__=='__main__':unittest.main()

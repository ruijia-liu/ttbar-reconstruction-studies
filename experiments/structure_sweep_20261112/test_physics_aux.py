"""Physical identities, gradients, source alignment and train-only scaling."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from physics_aux import (AUX_KEYS,MAIN_KEYS,P4_KEYS,p4_to_auxiliary,load_auxiliary,
                         fit_aux_stats,standardize_aux,reconstruct_observables)


def make_p4(momenta, masses):
    p=np.asarray(momenta,dtype=float).reshape(-1,2,3)
    m=np.broadcast_to(np.asarray(masses,dtype=float),p.shape[:2])
    return np.concatenate([p,np.sqrt(np.sum(p*p,axis=2)+m*m)[...,None]],axis=2).reshape(-1,8)


class PhysicsAuxTests(unittest.TestCase):
    def call(self,z,mean=None,std=None):
        return reconstruct_observables(z,torch.zeros(8) if mean is None else mean,
            torch.ones(8) if std is None else std,torch.zeros(4),torch.ones(4))

    def test_rest_parents_reconstruct_mass_sum_and_zero_rapidity(self):
        p4=make_p4(np.zeros((1,2,3)),[170.,175.])
        x,_=p4_to_auxiliary(p4)
        result=self.call(torch.tensor(x)).numpy()
        np.testing.assert_allclose(result,[[345.,0.]],rtol=3e-7,atol=1e-5)

    def test_back_to_back_mass_and_asymmetric_rapidity(self):
        p=np.array([[[40.,-30.,100.],[-40.,30.,-200.]]])
        masses=np.array([172.,174.]);p4=make_p4(p,masses)
        x,_=p4_to_auxiliary(p4);result=self.call(torch.tensor(x)).numpy()[0]
        pair=p4.reshape(1,2,4).sum(1)[0]
        expected_mass=np.sqrt(pair[3]**2-np.sum(pair[:3]**2))
        e=p4.reshape(1,2,4)[0,:,3]
        y=.5*np.log((e+p[0,:,2])/(e-p[0,:,2]))
        np.testing.assert_allclose(result,[expected_mass,abs(y[0])-abs(y[1])],rtol=5e-7,atol=1e-5)

    def test_gradients_match_finite_differences(self):
        raw=np.array([[40.,-30.,100.,np.log(172.),-40.,30.,-200.,np.log(174.)]])
        z=torch.tensor(raw,dtype=torch.float64,requires_grad=True)
        out=self.call(z);objective=out[0,0]/100+out[0,1]
        gradient=torch.autograd.grad(objective,z)[0].numpy()[0]
        # Finite difference physical values in float32 output needs a moderate step.
        for j in range(8):
            step=1e-2 if j%4!=3 else 1e-3
            plus=raw.copy();minus=raw.copy();plus[0,j]+=step;minus[0,j]-=step
            fp=self.call(torch.tensor(plus))[0];fm=self.call(torch.tensor(minus))[0]
            finite=float(((fp[0]-fm[0])/100+fp[1]-fm[1])/(2*step))
            self.assertAlmostEqual(gradient[j],finite,delta=max(2e-4,abs(gradient[j])*2e-3))

    def test_normalization_and_float32_bfloat16(self):
        raw,_=p4_to_auxiliary(make_p4([[[10.,20.,30.],[-10.,-20.,-30.]]],[172.,172.]))
        stats=fit_aux_stats(raw)
        z=torch.tensor(standardize_aux(raw,stats),dtype=torch.bfloat16,requires_grad=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result=reconstruct_observables(z,stats['mean'],stats['std'],[300.,.1,0,0],[100.,2,1,1])
        self.assertEqual(result.dtype,torch.float32)
        loss=result.sum();loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        expected=self.call(torch.tensor(raw))
        torch.testing.assert_close(result, (expected-torch.tensor([[300.,.1]]))/torch.tensor([[100.,2.]]),rtol=1e-5,atol=1e-5)

    def test_train_stats_floors_and_validation_does_not_refit(self):
        train=np.zeros((3,8),dtype=np.float32);train[:,[3,7]]=np.log(172.)
        stats=fit_aux_stats(train);before=dict(stats)
        val=train+100
        z=standardize_aux(val,stats)
        self.assertEqual(stats,before)
        np.testing.assert_allclose(stats['std'],[1,1,1,.05,1,1,1,.05])
        self.assertGreater(abs(z).mean(),100.)
        self.assertEqual(stats['fit_rows'],3)

    def test_nonphysical_truth_rejected_without_dropping_rows(self):
        for bad in [np.array([[1,0,0,.5,0,0,0,172.]]),np.array([[0,0,0,-172.,0,0,0,172.]])]:
            with self.assertRaisesRegex(ValueError,'no rows dropped'):p4_to_auxiliary(bad)
        with self.assertRaises(ValueError):p4_to_auxiliary(np.full((2,8),np.nan))

    def test_nonfinite_reconstruction_raises(self):
        z=torch.zeros(2,8);z[:,[3,7]]=1000
        with self.assertRaises(FloatingPointError):self.call(z)

    def test_parquet_nested_labels_keep_event_order(self):
        import awkward as ak
        p4=make_p4([[[0,0,0],[0,0,0]],[[10,20,30],[-10,-20,-30]],[[1,2,3],[4,5,6]]],[172.,173.])
        truth=np.arange(12,dtype=float).reshape(3,4)
        truth[:,:2]=self.call(torch.tensor(p4_to_auxiliary(p4)[0])).numpy()
        target={key:p4[:,j].copy() for j,key in enumerate(P4_KEYS)}
        target.update({key:truth[:,j].copy() for j,key in enumerate(MAIN_KEYS)})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'synthetic.parquet'
            ak.to_parquet(ak.Array(dict(targets=target,inputs=dict(dummy=np.arange(3)))),path)
            loaded=load_auxiliary(path,batch_size=2)
        np.testing.assert_array_equal(loaded['targets'],truth)
        np.testing.assert_array_equal(loaded['auxiliary'],p4_to_auxiliary(p4)[0])
        self.assertEqual(loaded['report']['rows_removed'],0)

    def test_inconsistent_main_physics_labels_are_rejected(self):
        import awkward as ak
        p4=make_p4(np.zeros((1,2,3)),[172.,173.])
        target={key:p4[:,j].copy() for j,key in enumerate(P4_KEYS)}
        target.update(dict(ttbar_mass=np.array([345.*1.01]),delta_abs_y=np.array([0.]),
                           cos_K_plus=np.array([0.]),cos_phi=np.array([0.])))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'inconsistent.parquet'
            ak.to_parquet(ak.Array(dict(targets=target)),path)
            with self.assertRaisesRegex(ValueError,'disagree with primary'):load_auxiliary(path)

if __name__=='__main__':unittest.main()

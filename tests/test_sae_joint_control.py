import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from length_budget_distill.sae_joint_control import JointRandomController

class StructuralControlTest(unittest.TestCase):
    def test_random_positive_and_negative_groups_both_act(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'sae.safetensors'
            save_file({'encoder_weight':torch.eye(4),'decoder_weight':torch.eye(4),
                'encoder_bias':torch.zeros(4),'decoder_bias':torch.zeros(4),
                'activation_mean':torch.zeros(4),'activation_scale':torch.ones(1)},str(p))
            layer=torch.nn.Identity()
            c=JointRandomController(positive_ids=[2],negative_ids=[3],short_ids=[0],long_ids=[1],random_ids=[2],
                torch_module=torch,checkpoint_path=p,layer_module=layer,k=4,maximum_delta_fraction=.3,
                device='cpu',dtype=torch.float32)
            c.begin(dict(mode='joint',count=1,rho=.3,start=0,structural_random=True),1)
            x=torch.ones(1,1,4);y=layer(x);c.end()
            self.assertTrue(torch.equal(y[...,:2],x[...,:2]))
            self.assertGreater(float(y[0,0,2]),1.)
            self.assertLess(float(y[0,0,3]),1.)
            self.assertAlmostEqual(float((y-x).norm()/x.norm()),.3,places=6)

if __name__=='__main__':unittest.main()

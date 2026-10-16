"""Analytic tests for local Jacobian operators and attribution-only gradients."""
import numpy as np
import torch
from tools.measure_sato_jacobian import LocalJacobian
from tools.audit_sato_gradient_connectivity import layerwise_attribution_audit


def test_jvp_vjp_and_power_match_known_linear_operator():
    matrix = torch.tensor([[3., 0.], [0., 2.], [0., 0.]], dtype=torch.float64)
    operator = LocalJacobian(lambda x: matrix @ x, torch.ones(2,dtype=torch.float64))
    assert torch.allclose(operator.jvp(torch.tensor([1.,2.],dtype=torch.float64)), matrix@torch.tensor([1.,2.],dtype=torch.float64))
    assert torch.allclose(operator.vjp(torch.ones(3,dtype=torch.float64)), matrix.T@torch.ones(3,dtype=torch.float64))
    spectral = operator.spectral(30,2,123)
    assert np.isclose(spectral['norm_estimate'],3.,rtol=1e-3)
    assert spectral['best_converged']
    # Diagonal J J^T gives an exact Hutchinson trace for every Rademacher probe.
    frob = operator.frobenius(8,123)
    assert np.isclose(frob['squared_norm'],13.)
    assert frob['probe_standard_error']==0


def test_scale_contraction_is_distinct_from_squared_frobenius_ratio():
    x=torch.ones(2,dtype=torch.float64)
    a=LocalJacobian(lambda x:3*x,x)
    b=LocalJacobian(lambda x:2.25*x,x)
    assert np.isclose(b.spectral(10,1,123)['norm_estimate']/a.spectral(10,1,123)['norm_estimate'],.75)
    assert np.isclose(b.frobenius(4,123)['squared_norm']/a.frobenius(4,123)['squared_norm'],.75**2)


def test_nonlinear_jvp_matches_central_difference():
    fn=lambda x:torch.stack([x[0].square()+x[1], torch.sin(x[1])])
    x=torch.tensor([.3,-.2],dtype=torch.float64)
    v=torch.tensor([.6,.8],dtype=torch.float64)
    operator=LocalJacobian(fn,x)
    central=(fn(x+1e-5*v)-fn(x-1e-5*v))/(2e-5)
    assert torch.allclose(operator.jvp(v),central,atol=1e-9)


def test_loss_sum_audit_marks_disconnected_not_connected_zero():
    generator=torch.nn.Module()
    generator.trans_base=torch.nn.Module()
    generator.trans_base.blocks=torch.nn.ModuleList([torch.nn.Linear(2,2)])
    generator.trans_head=torch.nn.Module()
    generator.trans_head.head=torch.nn.Linear(2,2)
    modules={'transformer':generator,'clip':torch.nn.Linear(2,2),
             'attr_pooling':torch.nn.Linear(2,2),'predictor':torch.nn.Linear(2,2)}
    for p in modules['clip'].parameters():p.requires_grad_(False)
    hidden=generator.trans_head.head(generator.trans_base.blocks[0](torch.ones(1,2)))
    attr_loss=modules['predictor'](modules['attr_pooling'](hidden.detach())).square().mean()
    result=layerwise_attribution_audit(attr_loss,modules)
    assert result['layers']['text_encoder']['status']=='frozen'
    assert result['layers']['trans_base.blocks.0']['status']=='disconnected'
    assert result['layers']['generation_head']['grad_l2_norm'] is None
    assert result['layers']['projection_head']['status']=='connected_nonzero'
    assert result['generator_gradient_l2']==0
    assert result['gradient_cosine_precondition']=='undefined_zero_generator_gradient'
    connected_loss=modules['predictor'](modules['attr_pooling'](hidden)).square().mean()
    result=layerwise_attribution_audit(connected_loss,modules)
    assert result['generator_gradient_l2']>0
    assert result['layers']['trans_base.blocks.0']['status']=='connected_nonzero'


def test_plot_axis_keeps_above_one_outliers_visible():
    from tools.analyze_sato_jacobian import ratio_axis_limits
    summary={'paired_rows':[{'raw':.5,'scaled':.75},{'raw':.9,'scaled':1.16}]}
    lo,hi=ratio_axis_limits(summary,['raw','scaled'])
    assert lo==0 and hi>1.16


def test_bootstrap_is_over_group_averages_and_keeps_all_values():
    from tools.analyze_sato_jacobian import paired_summary
    result=paired_summary([.5,.75,1.1],bootstrap=100,seed=123)
    assert np.isclose(result['mean'],(.5+.75+1.1)/3)
    assert result['median']==.75
    assert np.isclose(result['fraction_below_one'],2/3)
    assert result['n_groups']==3
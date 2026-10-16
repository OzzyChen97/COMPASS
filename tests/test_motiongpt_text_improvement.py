import pytest
from tools.eval_motiongpt_text_adaptation_validation import select_candidate


def rows(clean, p1, p2, p3):
    return [{'query':q,'lookup':'fallback_lower','R3':v} for q,v in zip(['clean','P1','P2','P3'],[clean,p1,p2,p3])]


def test_select_improvement_with_clean_guard():
    result=select_candidate({'original':rows(.8,.4,.5,.3),'previous_1e6':rows(.8,.41,.51,.31),
                             'lr1e5':rows(.79,.45,.55,.35),'lr3e5':rows(.75,.6,.7,.5)})
    assert result['selected']=='lr1e5'
    assert result['improvement_qualified']
    assert not result['scores']['lr3e5']['eligible']


def test_reject_single_level_regression_and_tiny_improvement():
    result=select_candidate({'original':rows(.8,.4,.5,.3),'previous_1e6':rows(.8,.41,.51,.31),
                             'lr1e5':rows(.8,.8,.8,.28),'lr3e5':rows(.8,.412,.512,.312)})
    assert result['selected']=='previous_1e6'
    assert not result['improvement_qualified']


def test_retain_original_if_old_adaptation_degrades_clean():
    result=select_candidate({'original':rows(.8,.4,.5,.3),'previous_1e6':rows(.7,.6,.7,.5),
                             'lr1e5':rows(.8,.4,.5,.3)})
    assert result['selected']=='original'


def test_reject_missing_or_invalid_validation_scores():
    with pytest.raises(ValueError):select_candidate({'original':rows(.8,.4,.5,.3)[:-1]})
    with pytest.raises(ValueError):select_candidate({'original':rows(float('nan'),.4,.5,.3)})
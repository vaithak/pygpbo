import pytest
from pygpbo import utils


def test_get_gpr_params():
    curr_dict = None
    res_dict  = utils.default_vals
    assert utils.get_gpr_params(curr_dict) == res_dict


    curr_dict = {
        'random_key': 'random_val'
    }
    res_dict['random_key'] = 'random_val'
    assert utils.get_gpr_params(curr_dict) == res_dict
    del res_dict['random_key']


    curr_dict = {
        'alpha': 1
    }
    res_dict['alpha'] = 1
    assert utils.get_gpr_params(curr_dict) == res_dict
    del res_dict['alpha']

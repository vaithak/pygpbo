# currently we support default values for only these parameters, 
# rest are the default ones used by sklearn
default_vals = {
    'alpha': 1e-6,
    'normalize_y': True,
    'n_restarts_optimizer': 5,
}


# set default values for the supported params if not already provided
def get_gpr_params(params):
    if params is None:
        params = {}

    for param, def_val in default_vals.items():
        if param not in params:
            params[param] = def_val

    return param


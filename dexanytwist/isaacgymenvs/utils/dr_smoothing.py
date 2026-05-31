import numpy as np
from copy import deepcopy

class DomainRandomizationSmoothing:
    def __init__(self, scale_initial, smoothing_begin, smoothing_end, smoothing_type, smoothing_factor):
        self.scale_initial = scale_initial
        self.smoothing_begin = smoothing_begin
        self.smoothing_end = smoothing_end
        self.smoothing_span = smoothing_end-smoothing_begin
        self.smoothing_type = smoothing_type
        self.smoothing_factor = smoothing_factor

    def linear_dr_smoothing(self, t):
        return min(1, self.scale_initial+t*((1-self.scale_initial)/(self.smoothing_span)))
    
    def root_dr_smoothing(self, t):
        lambda_p = self.scale_initial**self.smoothing_factor[0]
        return min(1, (t*(1-lambda_p)/(self.smoothing_span)+lambda_p)**(1/self.smoothing_factor[0]))
    
    def get_smoothing_scale(self, iter):
        t = max(0, iter-self.smoothing_begin)
        if self.smoothing_type=='linear':
            return self.linear_dr_smoothing(t)
        elif self.smoothing_type=='root':
            return self.root_dr_smoothing(t)

dr_params = {'frequency': 720, 'observations': {'range': [0, 0.005], 'range_correlated': [0, 0.01], 'operation': 'additive', 'distribution': 'gaussian'}, 'actions': {'range': [0.0, 0.05], 'range_correlated': [0, 0.02], 'operation': 'additive', 'distribution': 'gaussian'}, 'sim_params': {'gravity': {'range': [0, 0.4], 'operation': 'additive', 'distribution': 'gaussian'}}, 'actor_params': {'hand': {'color': False, 'dof_properties': {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform'}, 'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform'}, 'lower': {'range': [0, 0.01], 'operation': 'additive', 'distribution': 'gaussian'}, 'upper': {'range': [0, 0.01], 'operation': 'additive', 'distribution': 'gaussian'}}, 'rigid_body_properties': {'mass': {'range': [0.5, 1.5], 'operation': 'scaling', 'distribution': 'uniform', 'setup_only': True}}, 'rigid_shape_properties': {'friction': {'num_buckets': 250, 'range': [0.7, 1.3], 'operation': 'scaling', 'distribution': 'uniform'}, 'restitution': {'num_buckets': 100, 'range': [0.0, 0.4], 'operation': 'additive', 'distribution': 'uniform'}}}, 'object': {'scale': {'range': [0.8, 0.9], 'operation': 'scaling', 'distribution': 'uniform', 'setup_only': True}, 'rigid_body_properties': {'mass': {'range': [0.5, 1.5], 'operation': 'scaling', 'distribution': 'uniform', 'setup_only': True}}, 'rigid_shape_properties': {'friction': {'num_buckets': 250, 'range': [0.2, 1.3], 'operation': 'scaling', 'distribution': 'uniform'}, 'restitution': {'num_buckets': 100, 'range': [0.0, 0.4], 'operation': 'additive', 'distribution': 'uniform'}}}}}


def smoothing_params(dr_params, scale):
    dr_params_smoothed = deepcopy(dr_params)
    # scale = 0.1
    for nonphysical_param in ["observations", "actions"]:
        if nonphysical_param in dr_params:
            # print('input', dr_params_smoothed[nonphysical_param])
            # breakpoint()
            if dr_params_smoothed[nonphysical_param]['operation']=='additive':
                dr_params_smoothed[nonphysical_param]['range'] = [i*scale for i in dr_params_smoothed[nonphysical_param]['range']]
            elif dr_params_smoothed[nonphysical_param]['operation']=='scaling':
                dr_params_smoothed[nonphysical_param]['range'] = [(i-1)*scale + 1 for i in dr_params_smoothed[nonphysical_param]['range']]
            dr_params_smoothed[nonphysical_param]['range_correlated'] = [i*scale for i in dr_params_smoothed[nonphysical_param]['range_correlated']]

            # print(dr_params_smoothed[nonphysical_param])

        # print('-'*50)

    for actor, actor_properties in dr_params_smoothed["actor_params"].items():
        # print(f'      {actor}')
        for prop_name, prop_attrs in actor_properties.items():
            if prop_name == 'color':
                continue

            if prop_name == 'scale':
                setup_only = prop_attrs.get('setup_only', False)
                if not setup_only: 
                    raise NotImplementedError
                continue

            for p, og_p in prop_attrs.items():
                setup_only = og_p.get('setup_only', False)
                # print('-', p, og_p, setup_only)
                if setup_only: continue
                if p=='restitution' or p=='friction': continue
                # if p=='friction': continue
                if og_p['operation']=='additive':
                    og_p['range'] = [i*scale for i in og_p['range']]
                elif og_p['operation']=='scaling':
                    og_p['range'] = [(i-1)*scale + 1 for i in og_p['range']]
                # print('=', p, og_p, setup_only)
                
    return dr_params_smoothed

if __name__=="__main__":
    import matplotlib.pyplot as plt

    linear_dr_smoothing = DomainRandomizationSmoothing(0.0, 100, 400, 'linear', [])
    root_dr_smoothing = DomainRandomizationSmoothing(0.0, 100, 400, 'root', [3])
    iter_list = []
    linear_smoothing_scale_list = []
    root_smoothing_scale_list = []
    for iter in range(500):
        iter_list.append(iter)
        linear_smoothing_scale_list.append(linear_dr_smoothing.get_smoothing_scale(iter))
        root_smoothing_scale_list.append(root_dr_smoothing.get_smoothing_scale(iter))

    plt.plot(iter_list, linear_smoothing_scale_list)
    plt.plot(iter_list, root_smoothing_scale_list)
    plt.savefig('test.png')


    # dr_params_smoothed = smoothing_params(dr_params, 0.0)
    # print(dr_params_smoothed)

    # print('-'*50)

    # dr_params_smoothed = smoothing_params(dr_params, 0.2)
    # print(dr_params_smoothed)

    # print('-'*50)

    # dr_params_smoothed = smoothing_params(dr_params, 1.0)
    # print(dr_params_smoothed)

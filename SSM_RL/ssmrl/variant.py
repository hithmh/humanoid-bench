import gymnasium as gym
import humanoid_bench
from model_based_RL.replay_memory import *
SEED = None

VARIANT = {
    # 'env_name': 'FetchReach-v1',
    # 'env_name': 'Antcost-v0',
    # 'env_name': 'oscillator',
    # 'env_name': 'MJS1',
    # 'env_name': 'minitaur',
    # 'env_name': 'swimmer',
    # 'env_name': 'racecar',
    # 'env_name': 'MJS2',
    # 'env_name': 'oscillator_complicated',
    # 'env_name': 'waste_water',
    # 'env_name': 'waste_water_constant',
    # 'env_name': 'three_tank',
    # 'env_name': 'cartpole_cost',
    'env_name': 'h1-walk-v0',
    #training prams
    # 'algorithm_name': 'LAC',
    # 'algorithm_name': 'CEM_agent',
    # 'algorithm_name': 'DIOKO_agent_v2',
    #
    # 'algorithm_name': 'DIOKO_agent_v3',
    # 'algorithm_name': 'DIOKO_agent_v4',
    # 'algorithm_name': 'DIOKO_agent_v5',

    # 'algorithm_name': 'DIOKO_agent_v6',
    # 'algorithm_name': 'DIOKO_agent_v7',

    # 'algorithm_name': 'DIOKO_agent_v8',
    # 'algorithm_name': 'DIOKO_agent_v9',
    # 'algorithm_name': 'SSM_agent',
    # 'algorithm_name': 'SSM_agent_v2',
    # 'algorithm_name': 'SSM_agent_v3',
    # 'algorithm_name': 'SSM_agent_v4',
    # 'algorithm_name': 'SSM_agent_v5',
    # 'algorithm_name': 'SSM_agent_v6',
    # 'algorithm_name': 'SSM_agent_v7',
    # 'algorithm_name': 'SSM_agent_v8',
    'algorithm_name': 'SSM_agent_v9',

    # 'algorithm_name': 'DIOKO_agent_v11',

    # 'algorithm_name': 'SAC_cost',

    # 'algorithm_name': 'SPPO',
    # 'algorithm_name': 'DDPG',
    # 'algorithm_name': 'CPO',


    # 'additional_description': '-lr=1e-3_gamma=0.95',
    # 'additional_description': '-new_NoiseFree-lr=1e-3_bs=128',

    # 'additional_description': '-lr=1e-3-accelerated_NoiseFree',
    # 'additional_description': '-with_entropy_cons-1e3_train_steps',
    # 'additional_description': '-relu_QP-net=256-bs=128-latent=128-horizon=12-dicount=0.995-lr=5e-4-no_clip_sigma',
    # 'additional_description': '-pred_horizon=16-no_forward_pred-encoder_params_for_loss_P-target_entropy=20',
    # 'additional_description': '-predh=16-control_cost_fixed',
    # 'additional_description': '-sweep',
    'additional_description': '-AI_fixed-sweep',
    # 'additional_description': '-latent_fixed=4-bs64-forward_included-Hor4-lr_scale=0.8',

    # 'additional_description': '-pred=4-maximum_memory=1e5',


    # 'additional_description': '-epsilon=100_control-horizon=15',

    # 'additional_description': '-noisy-partial_terminal_cons-lr=1e-6-bs=256-gamma=0.95-clip_target',


    # 'evaluate': False,
    'train': True,
    # 'train': False,

    'num_of_trials': 10,   # number of random seeds
    'num_of_evaluation_paths': 3,  # number of rollouts for evaluation
    'eval_frequency_steps': 1000,  # number of steps between two evaluations
    'num_of_training_paths': 10,  # number of training rollouts stored for analysis
    'start_of_trial': 1,

    #evaluation params
    # 'evaluation_form': 'constant_impulse',
    'evaluation_form': 'dynamic',
    # 'evaluation_form': 'impulse',
    # 'evaluation_form': 'various_disturbance',
    # 'evaluation_form': 'param_variation',
    # 'evaluation_form': 'trained_disturber',
    'eval_list': [

        # 'DIOKO_agent_v11-lr=5e-7_train_steps=1e4'
        # 'DIOKO_agent_v10-lr=5e-7',
        # 'DIOKO_agent_v11-lr=5e-7-clip_lagrange-no_K-1e3',
        # 'SAC_cost-lr=1e-5',
        # 'DIOKO_agent_v11-lr=1e-6_weights=1-0.01_no-clip-on-sigma_sqrt',
        # 'CEM_agent-trial',
        # 'CEM_agent-accelerated_NoiseFree',
        # 'CEM_agent-new_NoiseFree',
        # 'SAC_cost-1e-5-longep',
        # 'SAC_cost-1e-5_static_input',
        # 'DIOKO_agent_v11-lr=1e-6_weights=1-0.01_no-clip-on-sigma_sqrt-longep',

        # 'SAC_cost-1e-5-longep_x0',
        'SAC_cost-1e-5_static_input_x0',
        # 'DIOKO_agent_v11-lr=1e-6_weights=1-0.01_no-clip-on-sigma_sqrt-longep_x0',

        ## CSTR
        # 'CEM_agent-lr=1e-4-accelerated_NoiseFree',
        # 'DIOKO_agent_v11-lr5e-6_weights=1-0.01_NoiseFree-bs=256-SCS_horizon=25',
        # 'SAC_cost-lr=1e-3',

    ],
    # 'trials_for_eval': [str(i) for i in range(1, 2)],
    # 'trials_for_eval': [str(i) for i in range(5, 6)],
    'trials_for_eval': [str(i) for i in range(0, 1)],

    'evaluation_frequency': 2048,
    'save_frequency': 2048
}
VARIANT['log_path']='/'.join(['./log', VARIANT['env_name'], VARIANT['algorithm_name'] + VARIANT['additional_description']])

ENV_PARAMS = {

    'h1hand-walk-v0':{
        'max_ep_steps': 1000,
        'max_global_steps': int(1e7),
        'max_episodes': int(1e7),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
        'apply_action_constraints': True,

    },
    'h1-walk-v0': {
        'max_ep_steps': 1000,
        'max_global_steps': int(1e7),
        'max_episodes': int(1e7),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
        'apply_action_constraints': True,

    },
    'three_tank': {
        'max_ep_steps': 2000,
        'max_global_steps': int(1e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,

        ### MPC params
        'reference': np.array([0.1763, 0.6731, 480.3165, 0.1965, 0.6536, 472.7863, 0.0651, 0.6703, 474.8877],
                              dtype=np.float32),
        'Q': np.diag([3., 3., 1., 3., 3., 1., 3, 3., 1.]),
        'R': np.diag(0.1 * np.ones([3])),

        'Qe': np.diag([0.005, 0.01, 1, 0.005, 0.01, 1, 0.005, 0.01, 1]),
        'Re': np.diag(0.01 * np.ones([3])),
        'end_weight': 100.,
        'control_horizon': 16,
        'control_prediction_horizon': 21,
        'MPC_pred_horizon': 25,
        'apply_state_constraints': False,
        'apply_action_constraints': True,

        'n_seeds': 500,
        'max_iters': 10,
        'elite_ratio': 0.1,
        'alpha': 0.1,
        'epsilon': 0.001,
        'network_structure':
            {'critic': [64, 64],
             'actor': [64, 64],
             },
    },

    'waste_water_constant': {
        'max_ep_steps':  5000,#1344
        'max_global_steps': int(1e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,

        'pred_dims_constrol': [0, 1],
        # 'pred_dims': [0, 1],
        'pred_dims': [i for i in range(0, 42)],
        # 'pred_dims': [i for i in range(0, 16)],
        ### MPC params
        'reference': np.array([1, 1],
                              dtype=np.float32),
        'Q': np.diag([1., 1.]),
        'R': np.diag(0.1 * np.ones([2])),
        # 'EMPC_R': np.array([8 / 1.8 / 1000 * 1333 * 0.3 / 100000, 0.004 * 0.3 / 100000]),
        'EMPC_R': np.array([8 / 1.8 / 1000 * 1333 * 0.3 / 100000, 0.004 * 0.3 / 100000]),
        # 'EMPC_R': np.array([0, 0]),
        'end_weight': 100.,
        'control_horizon': 16,
        'control_prediction_horizon': 16,
        'MPC_pred_horizon': 16,
        'apply_state_constraints': False,
        'apply_action_constraints': True,

        'n_seeds': 500,
        'max_iters': 10,
        'elite_ratio': 0.1,
        'alpha': 0.1,
        'epsilon': 0.001,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },

    },
    'cartpole_cost': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
        'apply_action_constraints': True,
        'control_horizon': 30,
        'control_prediction_horizon': 30,
    },
    'swimmer': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'oscillator': {
        'max_ep_steps': 400,
        'max_global_steps': int(1e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,
        'network_structure':
            {'critic': [256, 256, 16],
             'actor': [64, 64],
             },
    },
    'MJS1': {
        'max_ep_steps': 400,
        'max_global_steps': int(2e5),
        'max_episodes': int(2e5),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64,64],
             },
    },
    'MJS2': {
        'max_ep_steps': 400,
        'max_global_steps': int(2e5),
        'max_episodes': int(2e5),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64,64],
             },
    },
    'racecar': {
        'max_ep_steps': 20,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': True,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'minitaur': {
        'max_ep_steps': 500,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,
        # 'network_structure':
        #     {'critic': [98, 85, 16],
        #      'actor': [185,95],
        #      },
        'network_structure':
            {'critic': [256, 256, 16],
             'actor': [64,64],
             },
    },
    'oscillator_complicated': {
        'max_ep_steps': 400,
        'max_global_steps': int(1e5),
        'max_episodes': int(2e5),
        'disturbance dim': 2,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'HalfCheetahcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 6,
        'eval_render': False,
        'network_structure':
            {'critic': [256, 256, 16],
             'actor': [64, 64],
             },
    },
    'Quadrotorcost-v0': {
        'max_ep_steps': 2000,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'Antcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 8,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'FetchReach-v1': {
        # 'max_ep_steps': 50,
        'max_ep_steps': 200,
        'max_global_steps': int(3e5),
        'max_episodes': int(1e6),
        'disturbance dim': 4,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
}
ALG_PARAMS = {
    'MPC':{
        'horizon': 5,
    },
    'SSM_agent_v9': {
        'start_of_trial': 0,
        ### training hyperparameters
        'pretrain': False,
        'total_data_size': 2e3,
        # 'pretrain': False,
        # 'total_data_size': 0,
        'pretrain_steps': 2e4,
        'max_replay_steps': 1e6,

        'learning_rate': 3e-4,
        'encoder_lr_scale': 0.8,
        'decay_rate': 0.99,
        'decay_steps': 2e4,
        'steps_per_cycle': 1e2,
        'train_per_cycle': 1e2,

        'gamma': 0.995,
        'tau': 5e-3,
        'K_discount': 0.99,

        'target_entropy': 20,
        'activation': 'relu',
        # 'activation': 'elu',
        'encoder_struct': [512, 512],
        'policy_struct': [256, 256],
        'num_ensembles': 3,
        'latent_dim': 2**2,
        'pred_horizon': 4,
        'control_horizon': 4,
        'l2_regularizer': 1e-3,
        'val_frac': 0.01,
        'batch_size': 64,
        'history_horizon': 6,
    },
    'SSM_agent_v8': {
        'start_of_trial': 0,
        ### training hyperparameters
        'pretrain': True,
        'total_data_size': 2e3,
        # 'pretrain': False,
        # 'total_data_size': 0,
        'pretrain_steps': 2e4,

        'learning_rate': 5e-4,
        'decay_rate': 1,
        'decay_steps': 2,
        'steps_per_cycle': 1e3,
        'train_per_cycle': 1e3,

        'gamma': 0.995,
        'tau': 5e-3,
        'K_discount': 0.99,

        'target_entropy': 20,
        'activation': 'relu',
        # 'activation': 'elu',
        'encoder_struct': [256, 256],
        'policy_struct': [256, 256],
        'num_ensembles': 3,
        'latent_dim': 64,
        'pred_horizon': 16,
        'control_horizon': 16,
        'l2_regularizer': 0.01,
        'val_frac': 0.01,
        'batch_size': 64,
        'history_horizon': 6,
    },
    'SSM_agent_v7': {
        'start_of_trial': 0,
        ### training hyperparameters
        'pretrain': True,
        'total_data_size': 2e3,
        # 'pretrain': False,
        # 'total_data_size': 0,
        'pretrain_steps': 2e4,
        'max_replay_steps': 1e5,

        'learning_rate': 5e-4,
        'decay_rate': 1,
        'decay_steps': 2,
        'steps_per_cycle': 1e2,
        'train_per_cycle': 1e2,

        'gamma': 0.995,
        'tau': 5e-3,
        'K_discount': 0.99,

        'target_entropy': 20,
        'activation': 'relu',
        # 'activation': 'elu',
        'encoder_struct': [256, 256],
        'policy_struct': [256, 256],
        'num_ensembles': 3,
        'latent_dim': 2**10,
        'pred_horizon': 24,
        'control_horizon': 24,
        'l2_regularizer': 1e-3,
        'val_frac': 0.01,
        'batch_size': 64,
        'history_horizon': 6,
    },

    'SSM_agent_v6': {
        'start_of_trial': 0,
        ### training hyperparameters
        'pretrain': True,
        'total_data_size': 2e3,
        # 'pretrain': False,
        # 'total_data_size': 0,
        'pretrain_steps': 2e4,

        'learning_rate': 5e-5,
        'decay_rate': 0.99,
        'decay_steps': 2,
        'steps_per_cycle': 1e3,
        'train_per_cycle': 1e3,

        'gamma': 0.995,
        'tau': 1.,
        'K_discount': 0.99,

        'target_entropy': 20,
        'activation': 'relu',
        # 'activation': 'elu',
        'encoder_struct': [256, 256],
        'policy_struct': [256, 256],
        'num_ensembles': 3,
        'latent_dim': 64,
        'pred_horizon': 16,
        'control_horizon': 16,
        'l2_regularizer': 0.01,
        'val_frac': 0.01,
        'batch_size': 64,
        'history_horizon': 6,
    },
    'SSM_agent_v5': {
        'start_of_trial': 0,
        ### training hyperparameters
        'pretrain': True,
        'total_data_size': 2e3,
        # 'pretrain': False,
        # 'total_data_size': 0,
        'pretrain_steps': 2e4,

        'learning_rate': 5e-4,
        'decay_rate': 0.99,
        'decay_steps': 2,
        'steps_per_cycle': 1e3,
        'train_per_cycle': 1e3,

        'gamma': 0.995,
        'tau': 1.,
        'K_discount': 0.99,

        'target_entropy': 20,
        'activation': 'relu',
        # 'activation': 'elu',
        'encoder_struct': [256, 256],
        'policy_struct': [256, 256],
        'latent_dim': 64,
        'pred_horizon': 16,
        'control_horizon': 16,
        'l2_regularizer': 0.01,
        'val_frac': 0.01,
        'batch_size': 64,
        'history_horizon': 6,
    },
    'SSM_agent_v4': {
        'start_of_trial': 0,
        ### training hyperparameters
        'pretrain': True,
        'total_data_size': 2e3,
        # 'pretrain': False,
        # 'total_data_size': 0,
        'pretrain_steps': 2e4,

        'learning_rate': 5e-4,
        'decay_rate': 0.99,
        'decay_steps': 2,
        'steps_per_cycle': 1e3,
        'train_per_cycle': 1e3,

        'gamma': 0.995,
        'tau': 1.,
        'K_discount': 0.99,

        'target_entropy': 20,
        'activation': 'relu',
        # 'activation': 'elu',
        'encoder_struct': [256, 256],
        'policy_struct': [256, 256],
        'latent_dim': 64,
        'pred_horizon': 8,
        'control_horizon': 8,
        'l2_regularizer': 0.01,
        'val_frac': 0.01,
        'batch_size': 64,
        'history_horizon': 6,
    },
    'SSM_agent_v2': {
        'start_of_trial': 0,
        ### training hyperparameters
        'pretrain': False,
        'total_data_size': 2e3,
        # 'pretrain': False,
        # 'total_data_size': 0,
        'pretrain_steps': 2e4,

        'learning_rate': 5e-7,
        'decay_rate': 1,
        'decay_steps': 2,
        'steps_per_cycle': 1e3,
        'train_per_cycle': 1e3,

        'gamma': 0.995,
        'tau': 1.,
        'K_discount': 0.99,

        'target_entropy': 180,
        'activation': 'relu',
        # 'activation': 'elu',
        'encoder_struct': [256, 256],
        'policy_struct': [256, 256],
        'latent_dim': 128,
        'pred_horizon': 12,
        'control_horizon': 12,
        'l2_regularizer': 0.01,
        'val_frac': 0.01,
        'batch_size': 128,
        'history_horizon': 0,
    },
    'SSM_agent_v3': {
        'start_of_trial': 0,
        ### training hyperparameters
        'pretrain': False,
        'total_data_size': 2e3,
        # 'pretrain': False,
        # 'total_data_size': 0,
        'pretrain_steps': 2e4,
        'max_replay_steps': 1e5,

        'learning_rate': 5e-4,
        'decay_rate': 1,
        'decay_steps': 2,
        'steps_per_cycle': 1e3,
        'train_per_cycle': 1e3,

        'gamma': 0.995,
        'tau': 1.,
        'K_discount': 0.99,

        'target_entropy': 180,
        'activation': 'relu',
        # 'activation': 'elu',
        'encoder_struct': [256, 256],
        'policy_struct': [256, 256],
        'latent_dim': 2**10,
        'pred_horizon': 12,
        'control_horizon': 12,
        'l2_regularizer': 0.01,
        'val_frac': 1e-3,
        'batch_size': 128,
        'history_horizon': 6,
    },
    'SSM_agent': {
        'start_of_trial': 0,
        ### training hyperparameters
        'pretrain': True,
        'total_data_size': 2e3,
        # 'pretrain': False,
        # 'total_data_size': 0,
        'pretrain_steps': 2e4,

        'learning_rate': 5e-4,
        'decay_rate': 1,
        'decay_steps': 2,
        'steps_per_cycle': 1e3,
        'train_per_cycle': 1e3,

        'gamma': 0.995,
        'tau': 1.,
        'K_discount': 0.99,

        'target_entropy': -30,
        'activation': 'relu',
        # 'activation': 'elu',
        'encoder_struct': [256, 256],
        'policy_struct': [256, 256],
        'latent_dim': 128,
        'pred_horizon': 12,
        'control_horizon': 12,
        'l2_regularizer': 0.01,
        'val_frac': 0.01,
        'batch_size': 128,
        'history_horizon': 0,
    },
    'CEM_agent': {
        'start_of_trial': 0,
        ### training hyperparameters
        'pretrain': False,
        'pretrain_steps': 2e4,
        'total_data_size': 2e3,
        'learning_rate': 1e-3,
        'decay_rate': 0.9,
        'decay_steps': 10,
        'steps_per_cycle': 1e2,
        'train_per_cycle': 5e2,
        'ensemble_size': 5,
        'pred_horizon': 16,

        'val_frac': 0.1,
        'batch_size': 128,

        ### model hyperparameters
        'ensemble_forward': False,
        'obs_transfer': False,
        'keep_logvar': True,
        'weight_decay': [0.0001, 0.00025, 0.00025, 0.0005],   # regularization ratio
        # 'ensemble_forward': True,
        'activation': 'ReLU', # Type of activation function. Can be: 'relu', 'swish'
        'hidden_size' : 500,

        ### CEM hyperparameters
        'prop_mode': 'TS_inf',
        'num_elites': 50,
        'control_horizon': 15, #25, ## number of control steps
        'alpha': 0.1,  ## moving average of elite controls, 0.1 means 10% of new control is from old elite controls
        'epsilon': 1e-3, ## variance threshold for early stopping
        'max_iters': 6,   ## number of iterations of CEM
        'popsize':  500, ## population size

        # 'elite_ratio': 0.1, ## percentage of elite controls
        # 'end_weight': 100, ## weight of the last step in the cost function
        # 'warm_start': False, ## whether to use the last elite control as the initial control for the next step
    },

    'DIOKO_agent_v10': {
        'start_of_trial': 0,
        ### training hyperparameters
        'pretrain': True,
        'total_data_size': 2e3,
        # 'pretrain': False,
        # 'total_data_size': 0,
        'pretrain_steps': 2e4,

        'learning_rate': 5e-6,
        'decay_rate': 0.99,
        'decay_steps': 2,
        'steps_per_cycle': 1e3,
        'train_per_cycle': 1e3,

        'gamma': 0.95,
        'tau': 1.,
        'K_discount': 0.99,

        'target_entropy': -30,
        'activation': 'relu',
        # 'activation': 'elu',
        'encoder_struct': [128, 128],
        'latent_dim': 60,
        'pred_horizon': 16,
        'l2_regularizer': 0.01,
        'val_frac': 0.01,
        'batch_size': 128,
        'history_horizon': 0,
    },



    'LQR':{
        'use_Kalman': False,
    },

    'LAC': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 2.,
        'alpha3': .1,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        # 'gamma': 0.75,
        'steps_per_cycle': 100,
        'train_per_cycle': 80,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        'approx_value': True,
        'value_horizon': 2,
        # 'finite_horizon': True,
        'finite_horizon': False,
        'soft_predict_horizon': False,
        'target_entropy': None,
        'history_horizon': 0,  # 0 is using current state only
    },

    'DDPG': {
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha3': 0.001,
        'tau': 5e-3,
        'noise': 1.,
        'lr_a': 3e-4,
        'lr_c': 3e-4,
        'gamma': 0.99,
        'steps_per_cycle': 100,
        'train_per_cycle': 80,
        'history_horizon': 0,  # 0 is using current state only
        },
    'SAC_cost': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 128,
        'labda': 1.,
        'alpha': 1.,
        'alpha3': 0.5,
        'tau': 5e-3,
        'lr_a': 1e-3,
        'lr_c': 3e-3,
        'lr_l': 3e-3,
        'gamma': 0.95,
        # 'gamma': 0.75,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': False,
        'adaptive_alpha': True,
        'target_entropy': None,

    },
    # 'SPPO': {
    #     'batch_size':10000,
    #     'output_format':['csv'],
    #     'gae_lamda':0.95,
    #     'safety_gae_lamda':0.5,
    #     'labda': 1.,
    #     'number_of_trajectory':10,
    #     'alpha3': 0.1,
    #     'lr_c': 1e-3,
    #     'lr_a': 1e-4,
    #     'gamma': 0.995,
    #     'cliprange':0.2,
    #     'delta':0.01,
    #     'd_0': 1,
    #     'form_of_lyapunov': 'l_reward',
    #     'safety_threshold': 0.,
    #     'use_lyapunov': False,
    #     'use_adaptive_alpha3': False,
    #     'use_baseline':False,
    #     },
}


EVAL_PARAMS = {
    'param_variation': {
        'param_variables': {
            'mass_of_pole': np.arange(0.05, 0.55, 0.05),  # 0.1
            'length_of_pole': np.arange(0.1, 2.1, 0.1),  # 0.5
            'mass_of_cart': np.arange(0.1, 2.1, 0.1),    # 1.0

        },
        'grid_eval': True,
        # 'grid_eval': False,
        'grid_eval_param': ['length_of_pole', 'mass_of_cart'],
        'num_of_paths': 100,   # number of path for evaluation
    },
    'impulse': {
        # 'magnitude_range': np.arange(150, 160, 5),
        'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 10),
        # 'magnitude_range': np.arange(0.1, 1.1, .1),
        'num_of_paths': 100,   # number of path for evaluation
        'impulse_instant': 200,
    },
    'constant_impulse': {
        # 'magnitude_range': np.arange(120, 125, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(0.2, 2.2, .2),
        'magnitude_range': np.arange(0.1, 1.0, .1),
        'num_of_paths': 20,   # number of path for evaluation
        'impulse_instant': 20,
    },
    'various_disturbance': {
        'form': ['sin', 'tri_wave'][0],
        'period_list': np.arange(2, 11, 1),
        # 'magnitude': np.array([1, 1, 1, 1, 1, 1]),
        'magnitude': np.array([80]),
        # 'grid_eval': False,
        'num_of_paths': 100,   # number of path for evaluation
    },
    'trained_disturber': {
        # 'magnitude_range': np.arange(80, 125, 5),
        # 'path': './log/cartpole_cost/RLAC-full-noise-v2/0/',
        'path': './log/HalfCheetahcost-v0/RLAC-horizon=inf-dis=.1/0/',
        'num_of_paths': 100,   # number of path for evaluation
    },
    'dynamic': {
        'eval_additional_description': 'original',
        'num_of_paths': 1,   # number of path for evaluation
        # 'plot_average': True,
        'plot_average': False,
        'directly_show': True,
    },
}
VARIANT['env_params']=ENV_PARAMS[VARIANT['env_name']]
VARIANT['eval_params']=EVAL_PARAMS[VARIANT['evaluation_form']]
VARIANT['alg_params']=ALG_PARAMS[VARIANT['algorithm_name']]

for key in ENV_PARAMS[VARIANT['env_name']].keys():
    VARIANT[key] = ENV_PARAMS[VARIANT['env_name']][key]
for key in ALG_PARAMS[VARIANT['algorithm_name']].keys():
    VARIANT[key] = ALG_PARAMS[VARIANT['algorithm_name']][key]
for key in EVAL_PARAMS[VARIANT['evaluation_form']].keys():
    VARIANT[key] = EVAL_PARAMS[VARIANT['evaluation_form']][key]

RENDER = True
def get_env_from_name(name):
    if name == 'cartpole_cost':
        from envs.ENV_V1 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'waste_water':
        from envs.waste_water_system import waste_water_system as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'waste_water_constant':
        from envs.waste_water_system_constant import waste_water_system as dreamer
        env = dreamer()
        env = env.unwrapped
    else:
        env = gym.make(name)
        env = env.unwrapped
    if hasattr(env, 'seed') and SEED is not None:
        env.seed(SEED)
    return env

def get_train(name):
    if 'RARL' in name:
        from LAC.RARL import train as train
    elif 'LAC' in name:
        from LAC.LAC_V1 import train
    elif 'agent' in name:
        from model_based_RL.train import train
    else:
        from LAC.SAC_cost import train

    return train

def get_policy(name):
    if 'RARL' in name:
        from LAC.RARL import RARL as build_func
    elif 'LAC' in name :
        from LAC.LAC_V1 import LAC as build_func
    elif 'LQR' in name:
        from LAC.lqr import LQR as build_func
    elif 'MPC' in name:
        from LAC.MPC import MPC as build_func
    # elif 'CPO' in name:
    #     from CPO.CPO2 import CPO as build_func
    elif 'DDPG' in name:
        from LAC.SDDPG_V8 import SDDPG as build_func
    else:
        from LAC.SAC_cost import SAC_cost as build_func
    return build_func

def get_MBRL_agent(name):
    if name == 'CEM_agent':
        from model_based_RL.CEM_agent import CEM_RL_agent as build_func
    elif name == 'DIOKO_agent_v10':
        from model_based_RL.DIOKO_agent_v10 import DIOKO_agent as build_func
    elif name == 'SSM_agent':
        from model_based_RL.SSM_agent import SSM_agent as build_func
    elif name == 'SSM_agent_v2':
        from model_based_RL.SSM_agent_v2 import SSM_agent as build_func
    elif name == 'SSM_agent_v3':
        from model_based_RL.SSM_agent_v3 import SSM_agent as build_func
    elif name == 'SSM_agent_v4':
        from model_based_RL.SSM_agent_v4 import SSM_agent as build_func
    elif name == 'SSM_agent_v5':
        from model_based_RL.SSM_agent_v5 import SSM_agent as build_func
    elif name == 'SSM_agent_v6':
        from model_based_RL.SSM_agent_v6 import SSM_agent as build_func
    elif name == 'SSM_agent_v7':
        from model_based_RL.SSM_agent_v7 import SSM_agent as build_func
    elif name == 'SSM_agent_v8':
        from model_based_RL.SSM_agent_v8 import SSM_agent as build_func
    elif name == 'SSM_agent_v9':
        from model_based_RL.SSM_agent_v9 import SSM_agent as build_func
    else:
        raise ValueError(f"Unknown MBRL agent: {name}")

    return build_func
def get_eval(name):
    if 'LAC' in name or 'SAC_cost' in name:
        from LAC.LAC_V1 import eval

    return eval

def get_rm_func(name):
    list_of_past_memory = ['SSM_agent_v3', 'SSM_agent_v4', 'SSM_agent_v5', 'SSM_agent_v6', 'SSM_agent_v7', 'SSM_agent_v8', 'SSM_agent_v9']
    if name in list_of_past_memory:
        RM = ReplayMemoryWithPast
    else:
        RM = ReplayMemory

    return RM

def store_hyperparameters(path, args):
    np.save(path + "/hyperparameters.npy", args)
def restore_hyperparameters(path):
    args = np.load(path + "/hyperparameters.npy", allow_pickle=True).item()
    return args

def update_control_params(source_args, target_args):
    for key in ENV_PARAMS[source_args['env_name']].keys():
        target_args[key] = source_args[key]

    return target_args
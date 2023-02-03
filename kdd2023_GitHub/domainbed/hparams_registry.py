
import numpy as np
from domainbed.lib import misc
import os

def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed, test_envs, step):

    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):

        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))


    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))



    if algorithm in ['DANN', 'CDANN']:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    elif algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r:r.choice([0.05, 0.1, 0.5]))

    elif algorithm == 'MBDG_Reg' or algorithm == 'MBDG_DA':
        _hparam('mbdg_lam_dist', 1.0, lambda r: r.uniform(0.5, 10.0))

    elif algorithm == 'MBDG':
        _hparam('mbdg_dual_step_size', 0.05, lambda r: r.uniform(0.001, 0.1)) # 0.05
        _hparam('mbdg_fair_step_size', 0.05, lambda r: r.uniform(0.001, 0.1)) # 0.05
        _hparam('mbdg_gamma1', 0.025, lambda r: r.uniform(0.0001, 0.01)) # 0.01
        _hparam('mbdg_gamma2', 0.025, lambda r: r.uniform(0.0001, 0.01))

    elif algorithm == 'MBDG_3':
        _hparam('mbdg_dual_step_size', 0.05, lambda r: r.uniform(0.001, 0.1)) # 0.05
        _hparam('mbdg_fair_step_size', 0.05, lambda r: r.uniform(0.001, 0.1)) # 0.05
        _hparam('mbdg_gamma2', 0.025, lambda r: r.uniform(0.0001, 0.01))

    elif algorithm == 'MBDG_WF':
        _hparam('mbdg_dual_step_size', 0.05, lambda r: r.uniform(0.001, 0.1)) # 0.05
        _hparam('mbdg_fair_step_size', 0.05, lambda r: r.uniform(0.001, 0.1)) # 0.05
        _hparam('mbdg_gamma1', 0.025, lambda r: r.uniform(0.0001, 0.01)) # 0.01

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "Mixup":
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm == "MMD" or algorithm == "CORAL":
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "IGA":
        _hparam('penalty', 1000, lambda r: 10**r.uniform(1, 5))



    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))


    if dataset == 'NYPD':
        _hparam('batch_size', 1024, lambda r: 1024)
        _hparam('mlp_width', 32, lambda r: 32)
        _hparam('mlp_depth', 3, lambda r: 3)
        _hparam('mlp_dropout', 0., lambda r: 0.)
    else:
        _hparam('batch_size', 22, lambda r: 22)


    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    
    if algorithm in ['MBDG', 'MBDG_Reg', 'MBDG_DA', 'MBDA', 'MBDG_3', 'MBDG_WF']:
        if dataset == 'ColoredMNIST':
            model_root = './domainbed/munit/saved_models/colored_mnist'

        elif dataset == 'PACS':
            # model_root = './domainbed/munit/saved_models/pacs/new'
            model_root = './domainbed/munit/saved_models/pacs/'


        elif dataset == 'VLCS':
            model_root = './domainbed/munit/saved_models/vlcs/new'

        elif dataset == 'CCMNIST1':
            model_root = './domainbed/munit/saved_models/CCMNIST1/'

        elif dataset == 'FairFace':
            model_root = './domainbed/munit/saved_models/FairFace/'

        elif dataset == 'YFCC':
            model_root = './domainbed/munit/saved_models/YFCC/'

        elif dataset == 'NYPD':
            model_root = './domainbed/munit/saved_models/NYPD/'

        else:
            raise NotImplementedError(f'Dataset {dataset} not implemented for MBDG')

        # model_path = os.path.join(model_root,
        #     f'model-dom{"".join([str(e) for e in test_envs])}.pt')
        model_path1 = os.path.join(model_root,
                                  f'{"".join([str(e) for e in test_envs])}_step1.pt')
        model_path2 = os.path.join(model_root,
                                   f'{"".join([str(e) for e in test_envs])}_step2.pt')
        model_path3 = os.path.join(model_root,
                                   f'{"".join([str(e) for e in test_envs])}_cotrain_step1.pt')
        config_path = os.path.join(model_root, 'config.yaml')

        _hparam('step', step, lambda r: step)
        _hparam('mbdg_model_path1', model_path1, lambda r: model_path1)
        _hparam('mbdg_model_path2', model_path2, lambda r: model_path2)
        _hparam('mbdg_model_path3', model_path3, lambda r: model_path3)
        _hparam('mbdg_config_path', config_path, lambda r: config_path)

    return hparams


def default_hparams(algorithm, dataset, test_envs, step):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0, test_envs, step).items()}


def random_hparams(algorithm, dataset, seed, test_envs):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed, test_envs).items()}

import argparse
import os
import shutil

import numpy as np
import torch
# import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_topk_promt_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D

from graphbap.bapnet import BAPNet

def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    root_dir = './'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=root_dir+'/configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default=root_dir+'/logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    args = parser.parse_args()

    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree(root_dir + '/models', os.path.join(log_dir, 'models'))

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    logger.info('Loading dataset...')

    subsets = get_topk_promt_dataset(
        config=config.data,
        transform=transform,
    )

    topk_prompt = config.data.topk_prompt

    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, config.train.val_batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    logger.info('Building model...')

    net_cond = BAPNet(ckpt_path=config.net_cond.ckpt_path, hidden_nf=config.net_cond.hidden_dim).to(args.device)

    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)

    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            all_batch = next(train_iterator)
            all_batch = [b.to(args.device) for b in all_batch]
            assert len(all_batch) == topk_prompt + 1, "wrong value of topk_prompt"

            prompt_batch_2, prompt_batch_3 = None, None
            if topk_prompt == 1:
                batch, prompt_batch = all_batch
            elif topk_prompt == 2:
                batch, prompt_batch, prompt_batch_2 = all_batch
            elif topk_prompt == 3:
                batch, prompt_batch, prompt_batch_2, prompt_batch_3 = all_batch
            else:
                raise ValueError(topk_prompt)

            gt_protein_pos = batch.protein_pos

            results = model.get_diffusion_loss(
                net_cond=net_cond,
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,

                prompt_ligand_pos=prompt_batch.ligand_pos,
                prompt_ligand_v=prompt_batch.ligand_atom_feature_full,
                prompt_batch_ligand=prompt_batch.ligand_element_batch,

                prompt_ligand_pos_2=prompt_batch_2.ligand_pos if prompt_batch_2 is not None else None,
                prompt_ligand_v_2=prompt_batch_2.ligand_atom_feature_full if prompt_batch_2 is not None else None,
                prompt_batch_ligand_2=prompt_batch_2.ligand_element_batch if prompt_batch_2 is not None else None,

                prompt_ligand_pos_3=prompt_batch_3.ligand_pos if prompt_batch_3 is not None else None,
                prompt_ligand_v_3=prompt_batch_3.ligand_atom_feature_full if prompt_batch_3 is not None else None,
                prompt_batch_ligand_3=prompt_batch_3.ligand_element_batch if prompt_batch_3 is not None else None
            )
            loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )

    def validate(it):
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
        all_pred_v, all_true_v = [], []
        with torch.no_grad():
            model.eval()
            for all_batch in tqdm(val_loader, desc='Validate'):
                all_batch = [b.to(args.device) for b in all_batch]
                assert len(all_batch) == topk_prompt + 1, "wrong value of topk_prompt"

                prompt_batch_2, prompt_batch_3 = None, None
                if topk_prompt == 1:
                    batch, prompt_batch = all_batch
                elif topk_prompt == 2:
                    batch, prompt_batch, prompt_batch_2 = all_batch
                elif topk_prompt == 3:
                    batch, prompt_batch, prompt_batch_2, prompt_batch_3 = all_batch
                else:
                    raise ValueError(topk_prompt)

                batch_size = batch.num_graphs
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        net_cond=net_cond,
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,

                        prompt_ligand_pos=prompt_batch.ligand_pos,
                        prompt_ligand_v=prompt_batch.ligand_atom_feature_full,
                        prompt_batch_ligand=prompt_batch.ligand_element_batch,

                        prompt_ligand_pos_2=prompt_batch_2.ligand_pos if prompt_batch_2 is not None else None,
                        prompt_ligand_v_2=prompt_batch_2.ligand_atom_feature_full if prompt_batch_2 is not None else None,
                        prompt_batch_ligand_2=prompt_batch_2.ligand_element_batch if prompt_batch_2 is not None else None,

                        prompt_ligand_pos_3=prompt_batch_3.ligand_pos if prompt_batch_3 is not None else None,
                        prompt_ligand_v_3=prompt_batch_3.ligand_atom_feature_full if prompt_batch_3 is not None else None,
                        prompt_batch_ligand_3=prompt_batch_3.ligand_element_batch if prompt_batch_3 is not None else None,

                        time_step=time_step
                    )
                    loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc
            )
        )
        return avg_loss


    try:
        best_loss, best_iter = None, None
        for it in range(1, config.train.max_iters + 1):
            # with torch.autograd.detect_anomaly():
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')

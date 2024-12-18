from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import os
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
import tqdm
from openbabel import openbabel
from rdkit import RDLogger
from tqdm import tqdm
from models_consistency import *
from train_consistency import *
from _util_consistency import *
from rl_training_utils import *
import torch
from collections import defaultdict
from random import shuffle
from torch_geometric.data import Batch
import pdb
from torch_geometric.loader import DataLoader
import docking_posecheck_utils
from collections import OrderedDict
import pickle

RDLogger.DisableLog("rdApp.*")
openbabel.obErrorLog.SetOutputLevel(0)
openbabel.obErrorLog.StopLogging()
message_handler = openbabel.OBMessageHandler()
message_handler.SetOutputLevel(0)

# def load_data_create_pdbqt(cfg):
#     data_module = get_datamodule(
#         cfg.data_name, batch_size=1 
#     )
#     data_module.setup(stage="fit")
#     test_dataset = data_module.test_dataset
#     usable_testdataset = []
#     excluded_indexes = []
#     num_skipped = 0
#     pocket_pdbqt_paths = []
#     if not os.path.exists(os.path.join(cfg.pdb_dir, cfg.data_name)):
#         os.makedirs(os.path.join(cfg.pdb_dir, cfg.data_name))  

#     for i, data in enumerate(tqdm(test_dataset)):
#         if torch.any(data['ligand']['scaffold_mask']):
#             pdb_path = str(data['protein'].path)
#             pocket_pdbqt_path = os.path.join(cfg.pdb_dir, cfg.data_name, f'protein_{i}.pdbqt')
            
#             #skip if already exist
#             if os.path.exists(pocket_pdbqt_path):
#                 print(f"Skipping already processed file: {pocket_pdbqt_path}")
#                 usable_testdataset.append(data)
#                 pocket_pdbqt_paths.append(pocket_pdbqt_path)
#                 continue  

#             commands_prot = [
#                 'eval "$(conda shell.bash hook)"', 
#                 "conda activate mgltools",
#                 f"/home/ubuntu/anaconda3/envs/mgltools/bin/python /home/ubuntu/anaconda3/envs/mgltools/bin/prepare_receptor4.py -r {pdb_path} -o {pocket_pdbqt_path}",
#             ]
#             try:
#                 docking_utils._run_commands(commands_prot)
#                 usable_testdataset.append(data)
#                 pocket_pdbqt_paths.append(pocket_pdbqt_path)

#             except RuntimeError as e:
#                 print(f"Error processing {pdb_path}, attempting fix.")
#                 try:
#                     commands_fix = [
#                         "source /home/ubuntu/anaconda3/etc/profile.d/conda.sh",
#                         "conda activate diffhopp_rl",
#                         f"pdbfixer {pdb_path} --output {pdb_path[:-4]}_fixed.pdb"
#                     ]
#                     docking_utils._run_commands(commands_fix)
#                     commands_prot_fixed = [
#                         'eval "$(conda shell.bash hook)"', 
#                         "conda activate mgltools",
#                         f"/home/ubuntu/anaconda3/envs/mgltools/bin/python /home/ubuntu/anaconda3/envs/mgltools/bin/prepare_receptor4.py -r {pdb_path[:-4]}_fixed.pdb -o {pocket_pdbqt_path}",
#                     ]
#                     docking_utils._run_commands(commands_prot_fixed)
#                     usable_testdataset.append(data)
#                     pocket_pdbqt_paths.append(pocket_pdbqt_path)

#                 except RuntimeError as e:
#                     print(f"Failed to fix and process index {i}: {e}")
#                     excluded_indexes.append(i)
#                     num_skipped += 1
#         else:
#             excluded_indexes.append(i)
#             num_skipped += 1

#     assert len(usable_testdataset) == len(pocket_pdbqt_paths)
#     return usable_testdataset, pocket_pdbqt_paths


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

@dataclass
class TrainingConfig:
    model_config: None
    consistency_training: ConsistencyTraining_DiffHopp
    consistency_sampling: ConsistencySamplingAndEditing_DiffHopp
    lit_cm_config: LitConsistencyModelConfig
    seed: int = 42
    model_ckpt_path: str = "/home/ubuntu/kiwoong/diffusion-hopping/checkpoints/cm"
    resume_ckpt_path: Optional[str] = None
    device: Optional[int] = None
    check_val_every_n_epoch: Optional[int] = None

@hydra.main(
    version_base=None, config_path=".", config_name="config_rlcm_docking"
)
def main(cfg: DictConfig) -> None:
    # accelerate stuff
    if not os.path.exists(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)

    accelerator_config = ProjectConfiguration(
    project_dir=os.path.join(cfg.training_logdir, cfg.wandb_name),
    automatic_checkpoint_naming=True,
    total_limit=99999,
    )
    # usable_testdataset, pocket_pdbqt_paths = load_data_create_pdbqt(cfg)
    with open('./docking_testdataset_crossdocked.pkl', 'rb') as file:
        usable_testdataset = pickle.load(file)

    identifiers = [a['identifier'] for a in usable_testdataset]
    ref_scores = {a['identifier']:a['ligand']['ref_score'] for a in usable_testdataset}
    accelerator = Accelerator(
        log_with="wandb",
        project_config=accelerator_config,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
        * (cfg.num_inference_steps ),
        mixed_precision="fp16",
    )
    accelerator.init_trackers(
        project_name=cfg.wandb_project,
        config=OmegaConf.to_container(cfg),
        init_kwargs={"wandb": {"name": cfg.wandb_name, "entity": cfg.wandb_entity}},
    )

    model_config = SimpleNamespace(
        architecture=Architecture.GVP,
        seed=1,
        dataset_name="pdbbind_filtered",
        condition_on_fg=False,
        batch_size=1,
        T=cfg.num_inference_steps,
        lr=1e-4,
        num_layers=6,
        joint_features=128,
        hidden_features=256,
        edge_cutoff=(None, 5, 5),
    )

    student_model, _, teacher_model = get_consistency_models()

    config = TrainingConfig(
        model_config=model_config,
        consistency_training=ConsistencyTraining_DiffHopp(final_timesteps = cfg.num_inference_steps),
        consistency_sampling=ConsistencySamplingAndEditing_DiffHopp(final_timesteps = cfg.num_inference_steps),
        lit_cm_config=LitConsistencyModelConfig(),
        device=accelerator.device,
        check_val_every_n_epoch=300
    )

    rl_checkpoint = torch.load(cfg.rl_trained_path, map_location=lambda storage, loc: storage)

    new_state_dict = OrderedDict((key.replace('module.', ''), value) for key, value in rl_checkpoint['model'].items())

    model = LitConsistencyModel(
        config.consistency_training,
        config.consistency_sampling,
        student_model,
        teacher_model,
        # ema_student_model,
        config.lit_cm_config,
    )

    model = model.student_model
    model.load_state_dict(new_state_dict)
    model.to(accelerator.device)
    optimizer = torch.optim.Adam(model.consistency_model.estimator.parameters(), lr=cfg.lr)

    dataloader = inf_iterator(DataLoader(usable_testdataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size))
    optimizer, model.consistency_model.estimator, dataloader = accelerator.prepare(optimizer, model.consistency_model.estimator, dataloader)
    sampling_sigmas = list(reversed(karras_schedule(cfg.num_inference_steps, cfg.sigma_min, cfg.sigma_max, cfg.rho, accelerator.device)))
    logging_step = 0

    print("==== DATA LOADED ====")

    ############################# TRAINING LOOP #############################

    per_id_stat_tracking = PerIDStatTracker(cfg.buffer_size, cfg.min_count)

    for epoch in range(cfg.num_epochs):
        # sampling loop

        for itr in range(cfg.batches_per_epoch):
            samples = []

            for itr in tqdm(range(cfg.sample_iters),disable=not accelerator.is_local_main_process):
                batch = next(iter(dataloader)).to(accelerator.device)
                mask = model.consistency_model.get_mask(batch)
                batch = model.consistency_model.centered_complex(batch, mask)
                batch = model.consistency_model.normalize(batch)
             
                sample = sample_from_cm(model, batch, sampling_sigmas, mask, accelerator.device, cfg)
                sample["rewards"], sample['connectivity'], sample['qed'], sample['sa'], sample['docking_scores'] = torch.tensor(reward_function_docking(sample["generated_mols"], identifiers, ref_scores, cfg))
                del sample["generated_mols"]
                samples.append(sample)
            
            samples = {k: [s[k] for s in samples] for k in samples[0].keys()}
            
            samples["rewards"] = torch.cat(samples["rewards"])
            samples["connectivity"] = torch.cat(samples["connectivity"])
            samples["qed"] = torch.cat(samples["qed"])
            samples["sa"] = torch.cat(samples["sa"])
            samples["docking_scores"] = torch.cat(samples["docking_scores"])

            samples["log_probs"] = torch.cat(samples["log_probs"])
            samples["id"] = [ x for xs in samples["id"] for x in xs]
            samples["states"] = [ x for xs in samples["states"] for x in xs]
            samples["next_states"] = [ x for xs in samples["next_states"] for x in xs]
            samples["sigmas"] = torch.cat(samples["sigmas"])
            
            gathered_rewards = accelerator.gather(samples["rewards"].to(accelerator.device))
            gathered_connectivity = accelerator.gather(samples["connectivity"].to(accelerator.device))
            gathered_qed = accelerator.gather(samples["qed"].to(accelerator.device))
            gathered_sa = accelerator.gather(samples["sa"].to(accelerator.device))
            gathered_docking_scores = accelerator.gather(samples["docking_scores"].to(accelerator.device))
            metrics = {
                    "rewards": gathered_rewards,
                    "mean_reward": gathered_rewards.mean(),
                    "std_reward": gathered_rewards.std(),
                    "mean_connectivity": gathered_connectivity.mean(),
                    "mean_qed": gathered_qed.mean(),
                    "mean_sa": gathered_sa.mean(),       
                    "mean_docking_score": gathered_docking_scores.mean(),                    
                    "epoch": epoch,
                }   
            # accelerator.print(metrics)

            if cfg.log_to_wandb:
                accelerator.log(
                metrics,
                step=logging_step,
                )
            gathered_ids = accelerator.gather(ids_to_torch(samples["id"]).to(accelerator.device))

            # compute advantages per prompt
            gathered_advantages = per_id_stat_tracking.update(gathered_ids, gathered_rewards)

            # shard advantages
            samples["advantages"] = (
                torch.as_tensor(gathered_advantages)
                .reshape(accelerator.num_processes, -1)[accelerator.process_index].to(accelerator.device)
            )
            del samples["rewards"]
            del samples["id"]
            del samples['connectivity']
            del samples['qed']
            del samples['sa']
            del samples['docking_scores']

            ## DO BATCHING
            total_batch_size, num_timesteps = samples["log_probs"].shape

            perm = torch.randperm(total_batch_size)
            samples = {k: [v[i] for i in perm] if isinstance(v, list) else v[perm] for k, v in samples.items()}
            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps) for _ in range(total_batch_size)]
            )
            for key in [ # TODO: what to change here, need to figure out sample function first
                "sigmas",
                "log_probs",
            ]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size)[:, None],
                    perms,
                ]

            for key in [
                "states",
                "next_states",
            ]:
                # pdb.set_trace()
                for i in range(total_batch_size):
                    samples[key][i] = [samples[key][i][j] for j in perms[i]]

            # rebatch for training
            samples_batched = {k: reshape(v, [int(total_batch_size*num_timesteps / cfg.train_batch_size_per_gpu), cfg.train_batch_size_per_gpu]) for k, v in samples.items()}
            samples_batched = [ dict(zip(samples_batched, x)) for x in zip(*samples_batched.values()) ]
            info = defaultdict(list)

            for k in range(cfg.num_inner_epochs):
                model.train()
                for i, sample in tqdm(
                        enumerate(samples_batched),
                        desc=f"Training, {epoch}.{itr}",
                        disable=not accelerator.is_local_main_process,
                    ):
                    entry_states = [s[0].to(accelerator.device) for s in sample["states"]]
                    ## convert HeteroData into HeteroBatch
                    entry_states = Batch.from_data_list(entry_states)
                    mask = model.consistency_model.get_mask(entry_states)

                    for j in tqdm(
                            range(num_timesteps),
                            desc="Timestep",
                            disable=not accelerator.is_local_main_process,
                        ):
                        state = Batch.from_data_list([ s[j].to(accelerator.device) for s in sample["states"] ])
                        next_state = Batch.from_data_list([ s[j].to(accelerator.device) for s in sample["next_states"] ])
                        sigma = sample["sigmas"][:,j]
                        old_lp  = sample["log_probs"][:,j]
                        advantage = sample["advantages"]

                        with accelerator.accumulate(model.consistency_model):
                            
                            # denoise
                            x = state.clone()
                            
                            x["ligand"].x[mask], x["ligand"].pos[mask] = model_forward_wrapper_difsigma(model, state, mask, sigma.unsqueeze(-1), cfg.sigma_data, cfg.sigma_min)
                            # y["ligand"].y[mask], y["ligand"].pos[mask] = model_forward_wrapper_difsigma(model, state, mask, sigma.unsqueeze(-1), cfg.sigma_data, cfg.sigma_min)
                            x["ligand"].pos[mask] = util.centered_batch(
                                x["ligand"].pos[mask], x["ligand"].batch[mask], dim_size=x.num_graphs
                            )

                            lps = []
                            for i, sig in enumerate(sigma):
                                msk = next_state["ligand"].batch == i  # Create a mask for the batch index
                                # log_prob_x = log_normal_reduce(next_state["ligand"].x[msk], means=x["ligand"].x[msk], log_scales=0.5 * torch.log(sig**2 - cfg.sigma_min**2))
                                # log_prob_pos = log_normal_reduce(next_state["ligand"].pos[msk], means=x["ligand"].pos[msk], log_scales= 0.5 * torch.log(sig**2 - cfg.sigma_min**2))
                                log_prob_x = log_normal_reduce(next_state["ligand"].x[msk], means=x["ligand"].x[msk], log_scales=torch.log(sig))
                                log_prob_pos = log_normal_reduce(next_state["ligand"].pos[msk], means=x["ligand"].pos[msk], log_scales= torch.log(sig))

                                lps.append(log_prob_x + log_prob_pos)

                            log_prob = torch.stack(lps)
                            old_lp = log_prob.detach()
                            ratio = torch.exp(log_prob - old_lp)
                            
                            unclipped_loss = - advantage * ratio
                            clipped_loss = - advantage * torch.clamp(
                                ratio,
                                # 1.0 - training_config.clip_range,
                                # 1.0 + training_config.clip_range,
                                1.0 - cfg.clip_range,
                                1.0 + cfg.clip_range,
                            )

                            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                            info["approx_kl"].append(
                                0.5
                                * torch.mean((log_prob - old_lp) ** 2)
                            )
                            info["clipfrac"].append(
                                torch.mean(
                                    (
                                        torch.abs(ratio - 1.0) > cfg.clip_range
                                    ).float()
                                )
                            )
                            info["ratio"].append(torch.mean(ratio))
                            info["loss"].append(loss)

                            # backward pass
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(
                                    model.consistency_model.parameters(), cfg.max_grad_norm
                                )
                            optimizer.step()
                            optimizer.zero_grad()

                        if accelerator.sync_gradients:
                            # print("j::::::::::::::::", j)
                            # print("i+1::::::::::::::::", i+1)
                            assert (j == num_timesteps - 1) and (
                                i + 1
                            ) % cfg.gradient_accumulation_steps == 0
                            # log training-related stuff
                            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                            info = accelerator.reduce(info, reduction="mean")
                            accelerator.log(info)
                            logging_step += 1
                            print(logging_step)
                            info = defaultdict(list)

        # Save the model at regular intervals, not just every epoch
        if epoch % 1 == 0 and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            save_dir = os.path.join(cfg.ckpt_dir, f"{cfg.wandb_name}_epoch_{epoch}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'checkpoint.pth')
            try:
                accelerator.save({
                    "model": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, save_path)
                print(f"Model saved successfully at {save_path}")
            except Exception as e:
                print(f"Failed to save model: {str(e)}")

def setup_seed(seed):
            
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    
    main()

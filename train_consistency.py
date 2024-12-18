from lightning import  LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.nn import functional as F
from torch import Tensor, nn
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from typing import List
from typing import Any, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Draw
from _util_consistency import  get_datamodule, get_consistency_models, ema_decay_rate_schedule
from diffusion_hopping.model.consistency_lightning import ConsistencyDiffusionHoppingModel
from diffusion_hopping.model import util as util
from models_consistency import *
import argparse
from types import SimpleNamespace
import wandb

image_to_tensor = ToTensor()


def get_train_smiles_consistency(train_dataset):
    return [Chem.MolToSmiles(item["ligand"].ref) for item in train_dataset]

def _update_ema_weights(    
                            ema_weight_iter: Iterator[Tensor],
                            online_weight_iter: Iterator[Tensor],
                            ema_decay_rate: float,
                        ) -> None:
    for ema_weight, online_weight in zip(ema_weight_iter, online_weight_iter):
        if ema_weight.data is None:
            ema_weight.data.copy_(online_weight.data)
        else:
            ema_weight.data.lerp_(online_weight.data, 1.0 - ema_decay_rate)

def update_ema_model_(
    ema_model: nn.Module, online_model: nn.Module, ema_decay_rate: float
) -> nn.Module:
    """Updates weights of a moving average model with an online/source model.

    Parameters
    ----------
    ema_model : nn.Module
        Moving average model.
    online_model : nn.Module
        Online or source model.
    ema_decay_rate : float
        Parameter that controls by how much the moving average weights are changed.

    Returns
    -------
    nn.Module
        Updated moving average model.
    """
    # Update parameters
    _update_ema_weights(
        ema_model.parameters(), online_model.parameters(), ema_decay_rate
    )
    # Update buffers
    _update_ema_weights(ema_model.buffers(), online_model.buffers(), ema_decay_rate)

    return ema_model

def model_forward_wrapper_difsigma(
    model: nn.Module,
    x: Tensor,
    mask: Tensor,
    sigma: Tensor,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
    **kwargs: Any,
) -> Tensor:
    """Wrapper for the model call to ensure that the residual connection and scaling
    for the residual and output values are applied.

    Parameters
    ----------
    model : nn.Module
        Model to call.
    x : Tensor
        Input to the model, e.g: the noisy samples.
    sigma : Tensor
        Standard deviation of the noise. Normally referred to as t.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    **kwargs : Any
        Extra arguments to be passed during the model call.

    Returns
    -------
    Tensor
        Scaled output from the model with the residual connection applied.
    """
    c_skip = skip_scaling(sigma, sigma_data, sigma_min)
    c_out = output_scaling(sigma, sigma_data, sigma_min)
    c_skip_expanded = c_skip[x['ligand'].batch[mask]]
    c_out_expanded = c_out[x['ligand'].batch[mask]]
    x_final_ligand, pos_out_ligand = model.consistency_model.estimator(x, sigma, mask)
    ligand_features = x['ligand'].x[mask]
    ligand_positions = x['ligand'].pos[mask]
    x_final_ligand_ligand_mask = c_skip_expanded * ligand_features + c_out_expanded * x_final_ligand
    pos_out_ligand_ligand_mask = c_skip_expanded * ligand_positions + c_out_expanded * pos_out_ligand
    return x_final_ligand_ligand_mask, pos_out_ligand_ligand_mask

def model_forward_wrapper(
    model: nn.Module,
    x: Tensor,
    mask: Tensor,
    sigma: Tensor,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
    **kwargs: Any,
) -> Tensor:
    """Wrapper for the model call to ensure that the residual connection and scaling
    for the residual and output values are applied.

    Parameters
    ----------
    model : nn.Module
        Model to call.
    x : Tensor
        Input to the model, e.g: the noisy samples.
    sigma : Tensor
        Standard deviation of the noise. Normally referred to as t.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    **kwargs : Any
        Extra arguments to be passed during the model call.

    Returns
    -------
    Tensor
        Scaled output from the model with the residual connection applied.
    """
    c_skip = skip_scaling(sigma, sigma_data, sigma_min)
    c_out = output_scaling(sigma, sigma_data, sigma_min)
    x_final_ligand, pos_out_ligand = model.consistency_model.estimator(x, sigma, mask)
    ligand_features = x['ligand'].x[mask]
    ligand_positions = x['ligand'].pos[mask]
    x_final_ligand_ligand_mask = c_skip * ligand_features + c_out * x_final_ligand
    pos_out_ligand_ligand_mask = c_skip * ligand_positions + c_out * pos_out_ligand
    return x_final_ligand_ligand_mask, pos_out_ligand_ligand_mask


@dataclass
class LitConsistencyModelConfig:
    initial_ema_decay_rate: float = 0.95
    student_model_ema_decay_rate: float = 0.999943
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.995)
    # betas: Tuple[float, float] = (0.5, 0.999)
    # sample_every_n_steps: int = 500
    num_samples: int = 8
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    # sigma_max: float = 1.0
    # sigma_max: float = 10
    rho: float = 7.0
    # rho: float = 1.0
    sigma_data: float = 0.5
    initial_timesteps: int = 2
    final_timesteps: int = 150
    lr_patience: int = 100
    lr_cooldown: int = 100
    # lr_scheduler_start_factor: float = 1e-5
    # lr_scheduler_iters: int = 10_000

class LitConsistencyModel(LightningModule):
    def __init__(
        self,
        consistency_training: ConsistencyTraining_DiffHopp,
        consistency_sampling: ConsistencySamplingAndEditing_DiffHopp(),
        student_model: ConsistencyDiffusionHoppingModel,
        teacher_model: ConsistencyDiffusionHoppingModel,
        # ema_student_model: ConsistencyDiffusionHoppingModel,
        config: LitConsistencyModelConfig,
    ) -> None:
        super().__init__()

        self.consistency_training = consistency_training
        # print("final timesteps for training are: ", consistency_training.final_timesteps)
        self.consistency_sampling = consistency_sampling
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.config = config
        self.num_timesteps = self.consistency_training.initial_timesteps

        self.validation_metrics = None
        self.molecule_builder = MoleculeBuilder(include_invalid=True)
        self.next_analyse_samples = 50
        self._run_validation = False

        # Freeze teacher and EMA student models and set to eval mode
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        # for param in self.ema_student_model.parameters():
        #     param.requires_grad = False
        self.teacher_model = self.teacher_model.eval()
        # self.ema_student_model = self.ema_student_model.eval()
    
    def setup_metrics(self, train_smiles):
        self.validation_metrics = torch.nn.ModuleDict(
            {
                "Novelty": MolecularNovelty(train_smiles),
                "Validity": MolecularValidity(),
                "Connectivity": MolecularConnectivity(),
                "Lipinski": MolecularLipinski(),
                "LogP": MolecularLogP(),
                "QED": MolecularQEDValue(),
                "SAScore": MolecularSAScore(),
            }
        )


    def training_step(self, batch , batch_idx: int) -> None:

        batch_size = batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)
        x_output, pos_output = self.consistency_training(
            self.student_model,
            self.teacher_model,
            batch,
            self.global_step,
            self.trainer.max_steps,
        )
        self.num_timesteps = x_output.num_timesteps

        # batch_temp_pos_pred = batch['ligand'].pos.clone()
        # batch_temp_pos_target = batch['ligand'].pos.clone()
        # batch_temp_pos_pred[batch['ligand'].scaffold_mask] = pos_output.predicted
        # batch_temp_pos_target[batch['ligand'].scaffold_mask] = pos_output.target

        # pred_dismat=torch.cdist(batch_temp_pos_pred,batch_temp_pos_pred,compute_mode='donot_use_mm_for_euclid_dist')
        # target_dismat=torch.cdist(batch_temp_pos_target,batch_temp_pos_target,compute_mode='donot_use_mm_for_euclid_dist')

        x_loss = F.mse_loss(
            x_output.predicted, x_output.target)
        pos_loss = F.mse_loss(
            pos_output.predicted, pos_output.target)
        
        # dist_loss=F.mse_loss(pred_dismat,target_dismat)
        
        # loss = lpips_loss + overflow_loss
        # total_loss = x_loss + pos_loss + dist_loss
        total_loss = x_loss + pos_loss 

        self.log_dict(
            {
                "train_total_loss": total_loss,
                "x_loss": x_loss,
                "pos_loss":pos_loss,
                # "dist_loss":dist_loss,
                "num_timesteps": x_output.num_timesteps,
            },
        on_epoch=True,
        logger=True,
        prog_bar=True,
        batch_size= batch_size,
        sync_dist = True
        )
        return total_loss

    def on_train_batch_end(
        self, outputs: Any, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> None:
        # Update teacher model
        ema_decay_rate = ema_decay_rate_schedule(
            self.num_timesteps,
            self.config.initial_ema_decay_rate,
            self.consistency_training.initial_timesteps,
        )
        update_ema_model_(self.teacher_model, self.student_model, ema_decay_rate)
        self.log_dict({"ema_decay_rate": ema_decay_rate}, sync_dist = True)

        # # Update EMA student model
        # update_ema_model_(
        #     self.ema_student_model,
        #     self.student_model,
        #     self.config.student_model_ema_decay_rate,
        # )
        #TODO sample할거면 이거 틀어야함 
        # if (
        #     (self.global_step + 1) % self.config.sample_every_n_steps == 0
        # ) or self.global_step == 0:
        #     self.__sample_and_log_samples(batch, batch_idx)

        #TODO sample validation step에서 하기로 함


    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.student_model.parameters(), lr=self.config.lr, betas=self.config.betas
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='min',
            factor=0.9, patience=self.config.lr_patience,
            threshold=0.0001, threshold_mode='rel',
            cooldown=self.config.lr_cooldown,
            min_lr=1e-06, eps=1e-06
        )
        sched = {"optimizer": opt, "scheduler": { "scheduler": sched, "monitor": "train_total_loss"}}

        return sched
    
    def on_validation_epoch_start(self) -> None:
        self._run_validation = (self.current_epoch % self.next_analyse_samples) == 0 
     

    def validation_step(self, batch, batch_idx: int):
        # Depending on your data loading, you may need to adjust this line as done in training_step
        batch_size = batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)
        
        # Run the forward pass without consistency training logic if that's specific to training
        x_output, pos_output = self.consistency_training(
            self.student_model,
            self.teacher_model,
            batch,
            self.global_step,
            self.trainer.max_steps,
        )
        # batch_temp_pos_pred = batch['ligand'].pos.clone()
        # batch_temp_pos_target = batch['ligand'].pos.clone()
        # batch_temp_pos_pred[batch['ligand'].scaffold_mask] = pos_output.predicted
        # batch_temp_pos_target[batch['ligand'].scaffold_mask] = pos_output.target

        # pred_dismat=torch.cdist(batch_temp_pos_pred,batch_temp_pos_pred,compute_mode='donot_use_mm_for_euclid_dist')
        # target_dismat=torch.cdist(batch_temp_pos_target,batch_temp_pos_target,compute_mode='donot_use_mm_for_euclid_dist')
        # Calculate the losses similarly to the training_step
        x_loss = F.mse_loss(x_output.predicted, x_output.target)
        pos_loss = F.mse_loss(pos_output.predicted, pos_output.target)
        # dist_loss= F.mse_loss(pred_dismat,target_dismat)

        # total_loss = x_loss + pos_loss + dist_loss
        total_loss = x_loss + pos_loss 
        
        # Logging the validation loss and other relevant metrics but not backpropagating or optimizing
        self.log_dict({
            "val_total_loss": total_loss,
            "val_x_loss": x_loss,
            "val_pos_loss": pos_loss,
            # "val_dist_loss": dist_loss,
            # Optionally include num_timesteps if it's relevant for validation
            "num_timesteps": x_output.num_timesteps,
        }, on_epoch=True, logger=True, prog_bar=True, batch_size=batch_size, sync_dist = True)
        if self._run_validation:
            self.analyse_samples(batch, batch_idx)
        # Optionally return whatever you might need for validation_epoch_end hooks
        return total_loss
    
    @torch.no_grad()
    def analyse_samples(self, batch, batch_idx) -> None:
        with torch.no_grad():
            sampling_sigmas = karras_schedule(
                self.config.final_timesteps, self.config.sigma_min, self.config.sigma_max, self.config.rho, self.student_model.device
            )
            # if self.final_timesteps == 1:
            #     sampling_sigmas= reversed(sampling_sigmas)
            #     sampling_sigmas[-1] = 80
            # else:
            sampling_sigmas= reversed(sampling_sigmas)
            # sampling_sigmas[-1] += 1e-8  

            # print('sampling single sigma :', sampling_sigmas[-1])
            
            # for sampling_sigmas in sampling_sampling_sigmas:

            sample_results = self.consistency_sampling(self.student_model,batch,sampling_sigmas)
            final_output = sample_results[-1]
            molecules = self.student_model.molecule_builder(final_output)
            
            for k, metric in self.validation_metrics.items():
                metric(molecules)
                self.log(
                    f"{k}/val",
                    metric,
                    batch_size=batch.num_graphs,
                    sync_dist=True,
                )
            self.log_molecule_visualizations(molecules, batch_idx)

    def log_molecule_visualizations(self, molecules, batch_idx):
        images = []
        captions = []
        for i, mol in enumerate(molecules):
            if mol is None:
                continue
            img = image_to_tensor(Draw.MolToImage(mol, size=(500, 500)))
            images.append(img)
            captions.append(f"{self.current_epoch}_{batch_idx}_{i}")

        grid_image = make_grid(images)
        for logger in self.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_image(
                    f"log_image_{batch_idx}",
                    grid_image,
                    self.current_epoch,
                )
            if isinstance(logger, pl.loggers.WandbLogger):
                logger.log_image(key="test_set_images", images=images, caption=captions)

def get_callbacks(model_ckpt_path):
    # val_checkpoint = ModelCheckpoint(
    #     dirpath="/data/aigen/consistency/training/checkpoints/ver_dist_loss_gvp_150/val_best",
    #     filename="epoch={epoch}-step={step}-val_total_loss={val_total_loss:.3f}",
    #     monitor="val_total_loss",  # Changed from "loss/val" to "val_total_loss" as logged
    #     mode="min",
    #     auto_insert_metric_name=False,
    #     save_top_k=100,

    # )
    # train_checkpoint = ModelCheckpoint(
    #     dirpath="/data/aigen/consistency/training/checkpoints/ver_dist_loss_gvp_150/train_best",
    #     filename="epoch={epoch}-step={step}-train_total_loss={train_total_loss:.3f}",
    #     monitor="train_total_loss",  # Changed from "loss/val" to "val_total_loss" as logged
    #     mode="min",
    #     auto_insert_metric_name=False,
    #     save_top_k=10,

    # )
    # latest_checkpoint = ModelCheckpoint(
    #     dirpath="/data/aigen/consistency/training/checkpoints/ver_dist_loss_gvp_150/step_best",
    #     filename="latest-{epoch}-{step}",
    #     every_n_epochs=100,
    #     save_top_k= -1
        
    # )
    latest_checkpoint = ModelCheckpoint(
        dirpath=f"{model_ckpt_path}/step_best",
        filename="latest-{epoch}-{step}",
        every_n_epochs=50,
        save_top_k=-1
    )
    # return [val_checkpoint, train_checkpoint, latest_checkpoint]
    return [latest_checkpoint]



def run_training(config):

    seed_everything(config.seed)
        # Conditionally initialize wandb based on the wandb_logging flag
    if config.wandb_logging:
        os.environ['WANDB_DIR'] = '/data/aigen/consistency'
        run = wandb.init(project="diffusion_hopping_consistency")

    else:
        run = wandb.init(project="diffusion_hopping_consistency", mode="disabled")

    # run = wandb.init(project="diffusion_hopping_consistency", mode="disabled")
    student_model, _, teacher_model = get_consistency_models(T= config.lit_cm_config.final_timesteps)

    lit_cm = LitConsistencyModel(
        config.consistency_training,
        config.consistency_sampling,
        student_model,
        teacher_model,
        config.lit_cm_config,
    )
    
    wandb_logger = WandbLogger(experiment=run) 
    wandb_logger.watch(lit_cm)
    print("checkpoint path is: ", config.model_ckpt_path)
    trainer= Trainer(
        max_steps=1_000_000,
        precision="32-true",
        log_every_n_steps=1, 
        logger=wandb_logger,
        accelerator="gpu",
        callbacks = get_callbacks(config.model_ckpt_path),
        ###################################################################doing subset#########################
        # limit_val_batches=0.2,
        devices=config.device,
        # strategy='ddp_find_unused_parameters_true',
        strategy= 'ddp' if len(config.device)>1 else 'auto',
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        default_root_dir = config.model_ckpt_path)
        # max_epochs= config.max_epochs)

    data_module = get_datamodule(
    config.model_config.dataset_name, batch_size=config.model_config.batch_size // trainer.num_devices
    )
    data_module.setup(stage="fit")
    lit_cm.setup_metrics(get_train_smiles_consistency(data_module.train_dataset))
    trainer.fit(lit_cm, data_module.train_dataloader(), data_module.val_dataloader(), )

if __name__ == "__main__":
    disable_obabel_and_rdkit_logging()
    torch.set_float32_matmul_precision('medium')    
    parser = argparse.ArgumentParser(description='Training script for consistency models.')
    parser.add_argument('--ckpt_dir', type=str, default='/data/aigen/consistency/training/checkpoints', help='Base directory for saving checkpoints.')
    parser.add_argument('--final_timesteps', type=int, default=100, help='Number of final timesteps for the model.')
    parser.add_argument('--batch_size', type=int, default=20, help='Number of final timesteps for the model.')
    parser.add_argument('--devices', type=str, default='0', help='Comma-separated list of device indices for training, e.g., "0,1,2,3".')
    parser.add_argument('--wandb_logging', action='store_true', help='Enable WandB logging if set, otherwise disabled.')

    args = parser.parse_args()
    args.devices = [int(idx) for idx in args.devices.split(',')]

    model_config = SimpleNamespace(
    architecture=Architecture.GVP,
    seed=1,
    dataset_name="pdbbind_filtered",
    condition_on_fg=False,
    batch_size=args.batch_size,
    T=args.final_timesteps,
    lr=1e-3,
    num_layers=6,
    joint_features=128,
    hidden_features=256,
    edge_cutoff=(None, 5, 5),
    )
    model_config.attention = True

    @dataclass
    class TrainingConfig:
        # image_dm_config: ImageDataModuleConfig
        # unet_config: UNetConfig
        model_config: None
        consistency_training: ConsistencyTraining_DiffHopp
        consistency_sampling: ConsistencySamplingAndEditing_DiffHopp
        lit_cm_config: LitConsistencyModelConfig
        # trainer: Trainer
        seed: int = 42
        ckpt_dir: str = '/data/aigen/consistency/training/checkpoints'
        resume_ckpt_path: Optional[str] = None
        device: Optional[int] = None
        check_val_every_n_epoch: Optional[int] = None
        consistency_training: None
        consistency_sampling: None
        lit_cm_config: None
        check_val_every_n_epoch: int = 1
        wandb_logging: bool = False 

        def __post_init__(self):
            # Format current date
            current_date = datetime.now().strftime("%Y%m%d")
            # Update model_ckpt_path to include final_timesteps and date
            self.model_ckpt_path = f"{self.ckpt_dir}/{current_date}/ver_dist_loss_gvp_{self.lit_cm_config.final_timesteps}"
            # max_epochs: int = 1000
                                       
    config = TrainingConfig(
                        model_config=model_config,
                        # consistency_training=ConsistencyTraining_DiffHopp(final_timesteps=500),
                        # consistency_training=ConsistencyTraining_DiffHopp(),
                        consistency_training=ConsistencyTraining_DiffHopp(final_timesteps= args.final_timesteps),
                        consistency_sampling=ConsistencySamplingAndEditing_DiffHopp(final_timesteps= args.final_timesteps),
                        # consistency_sampling=ConsistencySamplingAndEditing_DiffHopp(),
                        # consistency_sampling=ConsistencySamplingAndEditing_DiffHopp(final_timesteps=500),
                        lit_cm_config=LitConsistencyModelConfig(
                        # lr_scheduler_iters=300000
                        final_timesteps= args.final_timesteps
                        ),
                        device=args.devices,
                        # device=[4,5,6,7],
                        # device=[1],
                        check_val_every_n_epoch=1,
                        wandb_logging=args.wandb_logging,
                        )
    run_training(config)
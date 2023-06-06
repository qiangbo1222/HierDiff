import imp
import logging
import math
from turtle import forward
from typing import Any, Dict

import torch
from hydra.utils import instantiate
from models.edge_denoise import Edge_denoise
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EdgeDenoise(LightningModule):
    #this is a training lopp 
    def __init__(self,cfg: Dict[str,Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        #self.automatic_optimization = False
        self.model = Edge_denoise(**cfg.model)
    
    def forward(self, batch):
        res = self.model(batch)
        return res 

    
    def training_step(self, batch,batch_idx):
        #opt = self.optimizers()
        #opt.zero_grad()
        result = self.forward(batch)
        loss = result["total_loss"]
        #self.manual_backward(loss)
        #opt.step()
        self.log('training_loss', loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log('training_focal_loss', result['focal_loss'], on_step=True, prog_bar=True, sync_dist=True)
        self.log('training_focal_accuracy', result['focal_accuracy'], on_step=True, prog_bar=True, sync_dist=True)
        self.log('training_edge_loss', result['edge_loss'], on_step=True, prog_bar=True, sync_dist=True)
        self.log('training_edge_accuracy', result['edge_accuracy'], on_step=True, prog_bar=True, sync_dist=True)
        self.log('training_node_loss', result['node_loss'], on_step=True, prog_bar=True, sync_dist=True)
        #self.log('perturb_loss', result['perturb_loss'], on_step=True, prog_bar=True, sync_dist=True)
        #self.log('training_node_size_loss', result['node_size_loss'], on_step=True, prog_bar=True, sync_dist=True)
        self.log('training_node_accuracy', result['node_accuracy'], on_step=True, prog_bar=True, sync_dist=True)
        return loss
        
    def training_epoch_end(self, result):
        sch = self.lr_schedulers()
        sch.step()

    
    def configure_optimizers(self):
        self.lr = self.cfg.optim.lr
        optimizer = instantiate(self.cfg.optim, self.model.parameters())
        scheduler = instantiate(
            self._set_num_training_steps(self.cfg.scheduler), optimizer
        )
        scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        print('complete config optimizers')
        return [optimizer], [scheduler]
    
    def _set_num_training_steps(self, scheduler_cfg):
        if "num_training_steps" in scheduler_cfg:
            scheduler_cfg = dict(scheduler_cfg)
            if self.global_rank == 0:
                logger.info("Computing number of training steps...")
                num_training_steps = [self.num_training_steps()]
            else:
                num_training_steps = [0]
            torch.distributed.broadcast_object_list(
                num_training_steps,
                0,
                group=torch.distributed.group.WORLD,
            )
            scheduler_cfg["num_training_steps"] = num_training_steps[0]
            logger.info(
                f"Training steps: {scheduler_cfg['num_training_steps']}"
            )
        return scheduler_cfg

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.num_training_batches != float("inf"):
            dataset_size = self.trainer.num_training_batches
        else:
            dataset_size = len(
                self.trainer._data_connector._train_dataloader_source.dataloader()
            )

        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches > 0
        ):
            dataset_size = min(dataset_size, self.trainer.limit_train_batches)
        else:
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)

        accelerator_connector = self.trainer._accelerator_connector
        if accelerator_connector.use_ddp2 or accelerator_connector.use_dp:
            effective_gpus = 1
        else:
            effective_gpus = self.trainer.devices
            if effective_gpus < 0:
                effective_gpus = torch.cuda.device_count()

        effective_devices = effective_gpus * self.trainer.num_nodes
        effective_batch_size = (
            self.trainer.accumulate_grad_batches * effective_devices
        )
        max_estimated_steps = (
            math.ceil(dataset_size // effective_batch_size)
            * self.trainer.max_epochs
        )
        logger.info(
            f"{max_estimated_steps} = {dataset_size} // "
            f"({effective_gpus} * {self.trainer.num_nodes} * "
            f"{self.trainer.accumulate_grad_batches}) "
            f"* {self.trainer.max_epochs}"
        )

        max_estimated_steps = (
            min(max_estimated_steps, self.trainer.max_steps)
            if self.trainer.max_steps and self.trainer.max_steps > 0
            else max_estimated_steps
        )
        return max_estimated_steps
    
    def _compute_metrics(self, result):
        #print(result.keys())
        # try:
        #     out_loss = result['CE_loss'].mean()
        #     print('CE_loss')
        # except:
        #     out_loss = result['loss'].mean()
        #     print("orig_loss")
        # mean_loss = result["loss"].mean()
        # ppl = torch.exp(out_loss)
        # print(ppl, mean_loss, out_loss)
        return {"loss":result["total_loss"].mean(),
                'focal_loss':result['focal_loss'].mean(),
                'edge_loss':result['edge_loss'].mean(),
                'node_loss':result['node_loss'].mean(),
                'focal_accuracy':result['focal_accuracy'].mean(),
                'edge_accuracy':result['edge_accuracy'].mean(),
                'node_accuracy':result['node_accuracy'].mean(),
                }#'perturb_loss':result['perturb_loss'].mean()
    
    def validation_step(self, batch,batch_idx):
        result = self.forward(batch)
        return result
    

    def validation_epoch_end(self, result):
        metrics = self._compute_metrics(self._gather_result(result))
        self.log('valid_loss', metrics['loss'], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid_focal_loss', metrics['focal_loss'], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid_edge_loss', metrics['edge_loss'], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid_node_loss', metrics['node_loss'], on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log('valid_perturb_loss', metrics['perturb_loss'], on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log('valid_node_size_loss', metrics['node_size_loss'], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid_focal_accuracy', metrics['focal_accuracy'], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid_edge_accuracy', metrics['edge_accuracy'], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid_node_accuracy', metrics['node_accuracy'], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid_all_accuracy', (metrics['focal_accuracy'] + metrics['edge_accuracy'] + metrics['node_accuracy']) / 3, on_epoch=True, prog_bar=True, sync_dist=True)
    
    
    def _gather_result(self, result):
        # collect steps
        result = {
            key: torch.cat([x[key] for x in result])
            if len(result[0][key].shape) > 0
            else torch.tensor([x[key] for x in result]).to(result[0][key])
            for key in result[0].keys()
        }
        # collect machines
        result = {
            key: torch.cat(list(self.all_gather(result[key])))
            for key in result.keys()
        }
        return result
    
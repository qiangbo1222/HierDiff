import imp
import logging
import math
from turtle import forward
from typing import Any, Dict

import torch
from hydra.utils import instantiate
from models.model_refine import Node2Vec
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Refine(LightningModule):
    #this is a training lopp 
    def __init__(self,cfg: Dict[str,Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        #self.automatic_optimization = False
        self.model = Node2Vec(**cfg.model)
    
    def forward(self, batch):
        res = self.model(batch)
        return res

    
    def training_step(self, batch, batch_idx):
        #opt = self.optimizers()
        #opt.zero_grad()
        result = self.forward(batch)
        loss = result["loss"]
        accuracy = result["accuracy"]
        #self.manual_backward(loss)
        #opt.step()
        self.log('training_loss', loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log('training_accuracy', accuracy, on_step=True, prog_bar=True, sync_dist=True)
        return loss
        
    def training_epoch_end(self, result):
        sch = self.lr_schedulers()
        sch.step()

    
    def configure_optimizers(self):
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
        return {"loss":result["loss"].mean(),
                "accuracy":result["accuracy"].mean()}
    
    def validation_step(self, batch,batch_idx):
        result = self.forward(batch)
        return result
    

    def validation_epoch_end(self, result):
        metrics = self._compute_metrics(self._gather_result(result))
        self.log(
            "val_loss",
            metrics["loss"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            "val_accuracy",
            metrics["accuracy"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        return metrics

    
    def test_step(self, batch, batch_nb):
        result = self.forward(batch)
        return result

    def test_epoch_end(self, result):
        result = self._gather_result(result)
        if self.global_rank == 0:
            self.log("test_loss", result["loss"], on_epoch=True, sync_dist=True)
    
    
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
    
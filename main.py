import hydra
from omegaconf import DictConfig, OmegaConf

from data.AlphazeroStyleDataModule import AlphazeroStyleDataModule
import omegaconf
import lightning as L

#TODO add overrides for things like vocabsize
@hydra.main(config_path="./conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    dl: AlphazeroStyleDataModule = hydra.utils.instantiate(cfg.data)
    dl.prepare_data()
    with omegaconf.open_dict(cfg):
        cfg.model.board_in_channels = dl.get_board_channels()
        cfg.model.eos_id = dl.sp.eos_id()
    model = hydra.utils.instantiate(cfg.model)
    trainer = L.Trainer(
        callbacks=[L.pytorch.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=False)],
        default_root_dir=cfg.data.artifacts_path
    )
    trainer.fit(model, dl)


if __name__ == '__main__':
    main()
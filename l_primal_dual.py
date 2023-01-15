from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path='./config/', config_name='train.yaml')
def cli_main(cfg: DictConfig) -> None:
    from train import train
    from util import extras
    extras(cfg)

    return train(cfg)

if __name__ == "__main__":
    cli_main()

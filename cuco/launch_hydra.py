#!/usr/bin/env python3
from pathlib import Path
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from cuco.core import EvolutionRunner


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    print("Experiment configurations:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    job_cfg = hydra.utils.instantiate(cfg.job_config)
    db_cfg = hydra.utils.instantiate(cfg.db_config)
    evo_cfg = hydra.utils.instantiate(cfg.evo_config)

    evo_runner = EvolutionRunner(
        evo_config=evo_cfg,
        job_config=job_cfg,
        db_config=db_cfg,
        verbose=cfg.verbose,
    )
    evo_runner.run()


if __name__ == "__main__":
    main()

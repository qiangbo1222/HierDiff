import json

import rich
from rich.syntax import Syntax
from rich.panel import Panel
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf


@rank_zero_only
def print_config(config: DictConfig, resolve: bool = True) -> None:
    content = OmegaConf.to_yaml(config, resolve=resolve)
    rich.print(Panel(
        Syntax(
            content, "yaml", background_color='default', line_numbers=True,
            code_width=80),
        title='Config', expand=False))


def save_lr_finder(lr_finder) -> None:
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_finder.png')
    lr_finder.results['suggestion'] = lr_finder.suggestion()
    with open('lr_finder.json', 'w') as f:
        json.dump(lr_finder.results, f, indent=2)

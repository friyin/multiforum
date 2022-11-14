#!/usr/bin/env python

import yaml
from pathlib import Path
config_data = yaml.safe_load(Path('config.yml').read_text())

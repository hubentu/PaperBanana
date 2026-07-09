# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration for experiments
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from utils.legacy_generation_options import is_plot_task


@dataclass
class ExpConfig:
    """Experiment configuration"""

    dataset_name: Literal["PaperBananaBench"]
    task_name: Literal["diagram", "plot"] = "diagram"
    split_name: str = "test"
    temperature: float = 1.0
    exp_mode: str = ""
    retrieval_setting: Literal["auto", "manual", "random", "none"] = "auto"
    planner_metaphor: bool = False
    max_critic_rounds: int = 3
    main_model_name: str = ""
    image_gen_model_name: str = ""
    work_dir: Path = Path(__file__).parent.parent
    # Optional override for Retriever/Planner refs. None → data/PaperBananaBench/{task}.
    ref_dir: Path | None = None

    timestamp: str | None = None

    def __post_init__(self):
        self.task_name = "plot" if is_plot_task(self.task_name) else "diagram"
        if self.ref_dir is not None:
            self.ref_dir = Path(self.ref_dir)
        os.environ["TZ"] = "America/Los_Angeles"  # set the timezone as you like
        if hasattr(time, "tzset"):
            time.tzset()  # Only available on Unix; no-op guard for Windows
        
        # Fallback to yaml config if no model name provided
        if not self.main_model_name or not self.image_gen_model_name:
            import yaml
            config_path = self.work_dir / "configs" / "model_config.yaml"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    model_config_data = yaml.safe_load(f) or {}
                    if not self.main_model_name:
                        self.main_model_name = model_config_data.get("defaults", {}).get("main_model_name", "")
                    if not self.image_gen_model_name:
                        self.image_gen_model_name = model_config_data.get("defaults", {}).get("image_gen_model_name", "")
        # Fallback to environment variables
        if not self.main_model_name:
            self.main_model_name = os.environ.get("MAIN_MODEL_NAME", "")
        if not self.image_gen_model_name:
            self.image_gen_model_name = os.environ.get("IMAGE_GEN_MODEL_NAME", "")
        # Hard defaults so model name is never empty
        if not self.main_model_name:
            self.main_model_name = "gemini-3.1-pro-preview"
            print(f"Warning: main_model_name not configured, falling back to '{self.main_model_name}'. "
                  "Set it in configs/model_config.yaml or via --main-model-name.")
        if not self.image_gen_model_name:
            self.image_gen_model_name = "gemini-3.1-flash-image-preview"
            print(f"Warning: image_gen_model_name not configured, falling back to '{self.image_gen_model_name}'. "
                  "Set it in configs/model_config.yaml or via --image-gen-model-name.")
        self.timestamp = (
            time.strftime("%m%d_%H%M") if self.timestamp is None else self.timestamp
        )
        self.exp_name = f"{self.timestamp}_{self.retrieval_setting}ret_{self.exp_mode}_{self.split_name}"

        # mkdir result_dir if not exists (skip on read-only FS, e.g. cwltool --read-only)
        self.result_dir = self.work_dir / "results" / f"{self.dataset_name}_{self.task_name}"
        try:
            self.result_dir.mkdir(exist_ok=True, parents=True)
        except OSError as e:
            # ponytail: CWL/docker --read-only can't create /app/results; callers use --output instead
            print(f"Warning: could not create result_dir {self.result_dir}: {e}")

    def resolve_ref_dir(self) -> Path:
        """Directory with ref.json and reference images for Retriever/Planner."""
        if self.ref_dir is not None:
            return self.ref_dir
        return self.work_dir / "data" / "PaperBananaBench" / self.task_name

    def resolve_ref_json(self) -> Path:
        return self.resolve_ref_dir() / "ref.json"

    def resolve_manual_ref_json(self) -> Path:
        return self.resolve_ref_dir() / "agent_selected_12.json"

    def resolve_ref_image(self, path_to_gt_image: str) -> Path:
        image_path = Path(path_to_gt_image)
        if image_path.is_absolute():
            return image_path
        return self.resolve_ref_dir() / image_path

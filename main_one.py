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
Single-sample PaperVizAgent entry point.

Accepts diagram or plot inputs directly (no dataset JSON file).
"""

import argparse
import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path

import aiofiles
from PIL import Image

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.polish_agent import PolishAgent
from agents.retriever_agent import RetrieverAgent
from agents.stylist_agent import StylistAgent
from agents.vanilla_agent import VanillaAgent
from agents.visualizer_agent import VisualizerAgent
from utils import config, paperviz_processor
from utils.legacy_generation_options import (
    generation_additional_info,
    normalize_legacy_input_content,
)


WORK_DIR = Path(__file__).parent
DEFAULT_MODEL_CONFIG = WORK_DIR / "configs" / "model_config.yaml"


def extract_final_image_b64(result: dict, task_name: str, exp_mode: str) -> tuple[str | None, str | None]:
    """Return (base64, field_name) for the best available generated image."""
    eval_key = result.get("eval_image_field")
    if eval_key and result.get(eval_key):
        return result[eval_key], eval_key

    for round_idx in range(3, -1, -1):
        key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        if result.get(key):
            return result[key], key

    fallbacks = [
        f"target_{task_name}_stylist_desc0_base64_jpg",
        f"target_{task_name}_desc0_base64_jpg",
        f"vanilla_{task_name}_base64_jpg",
        f"polished_{task_name}_base64_jpg",
    ]
    if "demo_full" not in exp_mode and "dev_full" not in exp_mode:
        fallbacks = [
            f"target_{task_name}_desc0_base64_jpg",
            f"target_{task_name}_stylist_desc0_base64_jpg",
            f"vanilla_{task_name}_base64_jpg",
            f"polished_{task_name}_base64_jpg",
        ]
    for key in fallbacks:
        if result.get(key):
            return result[key], key
    return None, None


def save_image_from_b64(b64: str, image_path: Path) -> Path:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img = Image.open(BytesIO(base64.b64decode(b64)))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "PNG" if image_path.suffix.lower() == ".png" else "JPEG"
    if fmt == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(image_path, format=fmt)
    return image_path


def apply_model_config(model_config_path: Path) -> dict:
    """Load YAML config and wire it into generation_utils API clients."""
    import yaml
    from utils import generation_utils

    if not model_config_path.exists():
        raise FileNotFoundError(f"model_config not found: {model_config_path}")

    with open(model_config_path, "r", encoding="utf-8-sig") as f:
        model_config_data = yaml.safe_load(f) or {}

    generation_utils.model_config = model_config_data
    generation_utils.reinitialize_clients()
    return model_config_data


def load_raw_text(args: argparse.Namespace) -> str:
    if args.raw_file:
        return Path(args.raw_file).read_text(encoding="utf-8")
    return args.raw_text


def build_input_data(
    task_name: str,
    raw_text: str,
    caption_intent: str,
    max_critic_rounds: int,
    aspect_ratio: str = "1:1",
    image_size: str = "1k",
) -> dict:
    content = normalize_legacy_input_content(raw_text, task_name) if task_name == "plot" else raw_text
    return {
        "filename": "one_input",
        "caption": caption_intent,
        "content": content,
        "visual_intent": caption_intent,
        "max_critic_rounds": max_critic_rounds,
        "additional_info": generation_additional_info(aspect_ratio, image_size),
    }


def resolve_model_names(model_config_data: dict, args: argparse.Namespace) -> tuple[str, str]:
    import os

    defaults = model_config_data.get("defaults", {})
    main_model_name = (
        args.main_model_name
        or defaults.get("main_model_name", "")
        or os.environ.get("MAIN_MODEL_NAME", "")
    )
    image_gen_model_name = (
        args.image_gen_model_name
        or defaults.get("image_gen_model_name", "")
        or os.environ.get("IMAGE_GEN_MODEL_NAME", "")
    )
    return main_model_name, image_gen_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PaperVizAgent single-sample processing (direct text input)"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="diagram",
        choices=["diagram", "plot"],
        help="task type: diagram or plot (default: diagram)",
    )
    parser.add_argument(
        "--raw_text",
        type=str,
        default="",
        help="methodology text (diagram) or raw plot data as text/JSON (plot)",
    )
    parser.add_argument(
        "--raw_file",
        type=Path,
        default=None,
        help="file containing raw_text (diagram methodology or plot raw data)",
    )
    parser.add_argument(
        "--caption_intent",
        type=str,
        default="",
        help="figure caption (diagram) or visual intent (plot)",
    )
    parser.add_argument(
        "--model_config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help=f"path to model_config.yaml (default: {DEFAULT_MODEL_CONFIG})",
    )
    parser.add_argument(
        "--exp_mode",
        type=str,
        default="demo_full",
        help="experiment pipeline mode (default: demo_full)",
    )
    parser.add_argument(
        "--retrieval_setting",
        type=str,
        default="auto",
        choices=["auto", "manual", "random", "none"],
        help="retrieval setting for planner agent (default: auto)",
    )
    parser.add_argument(
        "--ref_dir",
        type=Path,
        default=None,
        help=(
            "directory with ref.json and reference images for Retriever/Planner "
            "(default: data/PaperBananaBench/{task_name}; falls back to none if missing)"
        ),
    )
    parser.add_argument(
        "--planner-metaphor",
        action="store_true",
        help="enable diagram-only Planner visual-metaphor discovery before detailed description output",
    )
    parser.add_argument(
        "--max_critic_rounds",
        type=int,
        default=3,
        help="maximum number of critic rounds (default: 3)",
    )
    parser.add_argument(
        "--main_model_name",
        type=str,
        default="",
        help='main model name (default: from model_config "defaults.main_model_name")',
    )
    parser.add_argument(
        "--image_gen_model_name",
        type=str,
        default="",
        help='image generation model name (default: from model_config "defaults.image_gen_model_name")',
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="1:1",
        help="output image aspect ratio, e.g. 1:1, 16:9, 21:9 (default: 1:1)",
    )
    parser.add_argument(
        "--image_size",
        type=str,
        default="1k",
        choices=["1k", "2k", "4k"],
        help="output image resolution for providers that support it (default: 1k)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=".",
        help="output JSON path or directory (default: current folder → {exp_name}.json)",
    )
    parser.add_argument(
        "--image_output",
        type=Path,
        default=None,
        help="output image path (default: same stem as --output with .jpg)",
    )

    args = parser.parse_args()

    if not args.caption_intent.strip():
        parser.error("--caption_intent is required")
    if not args.raw_file and not args.raw_text.strip():
        parser.error("one of --raw_text or --raw_file is required")
    if args.ref_dir is not None:
        args.ref_dir = args.ref_dir.resolve()
        if not (args.ref_dir / "ref.json").exists():
            parser.error(f"--ref_dir must contain ref.json: {args.ref_dir / 'ref.json'}")

    return args


async def main() -> None:
    args = parse_args()
    task_name = "plot" if args.task_name == "plot" else "diagram"

    model_config_data = apply_model_config(args.model_config.resolve())
    main_model_name, image_gen_model_name = resolve_model_names(model_config_data, args)

    exp_config = config.ExpConfig(
        dataset_name="PaperBananaBench",
        task_name=task_name,
        split_name="one",
        exp_mode=args.exp_mode,
        retrieval_setting=args.retrieval_setting,
        planner_metaphor=args.planner_metaphor,
        max_critic_rounds=args.max_critic_rounds,
        main_model_name=main_model_name,
        image_gen_model_name=image_gen_model_name,
        work_dir=WORK_DIR,
        ref_dir=args.ref_dir,
    )

    output_filename = Path(args.output)
    # ponytail: --output . (or an existing dir) → write exp_name.json inside it
    if output_filename.exists() and output_filename.is_dir():
        output_filename = output_filename / f"{exp_config.exp_name}.json"
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    data = build_input_data(
        task_name,
        load_raw_text(args),
        args.caption_intent,
        args.max_critic_rounds,
        aspect_ratio=args.aspect_ratio,
        image_size=args.image_size,
    )
    print(f"Task: {task_name}")
    print(f"Model config: {args.model_config.resolve()}")
    if args.ref_dir:
        print(f"Ref dir: {args.ref_dir}")
    print(f"Aspect/size: {args.aspect_ratio} / {args.image_size}")
    print(f"Output file: {output_filename}")

    processor = paperviz_processor.PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    result_data = None
    async for item in processor.process_queries_batch([data], max_concurrent=1, do_eval=False):
        result_data = item

    if result_data is None:
        raise RuntimeError("Pipeline returned no result.")

    print(f"Saving result to {output_filename}")
    async with aiofiles.open(output_filename, "w", encoding="utf-8", errors="surrogateescape") as f:
        json_string = json.dumps([result_data], ensure_ascii=False, indent=4)
        json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
        await f.write(json_string)

    b64, image_key = extract_final_image_b64(result_data, task_name, args.exp_mode)
    if not b64:
        raise RuntimeError("Pipeline produced no image to save.")

    if args.image_output is not None:
        image_path = Path(args.image_output)
        if image_path.exists() and image_path.is_dir():
            image_path = image_path / f"{output_filename.stem}.jpg"
    else:
        image_path = output_filename.with_suffix(".jpg")

    save_image_from_b64(b64, image_path)
    print(f"Saved image to {image_path} (from {image_key})")
    print("Processing completed.")


if __name__ == "__main__":
    asyncio.run(main())

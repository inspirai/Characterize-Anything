import PIL
from tqdm import tqdm

from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
from inpainter.base_inpainter import BaseInpainter
import numpy as np
import argparse
import re
from role_play import generate_self_play_conversation


def format_conversation(conversation):
    # use regex to extract speaker and content
    pattern = r"([^\n:]+): ([^\n]+)"
    matches = re.findall(pattern, conversation)

    # convert speaker and content to specified format
    dialogue_list = [(0 if m[0] == "Coca-Cola Can" else 1, m[1]) for m in matches]

    print(dialogue_list)
    return dialogue_list


def format_generated_conversation(dialogue):
    # use regex to extract speaker and content
    pattern = r"\((\d+)\)：(.*?)\n"
    matches = re.findall(pattern, dialogue)
    result = [(int(match[0]), match[1]) for match in matches]
    return result


class TrackingAnything:
    def __init__(self, sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.e2fgvi_checkpoint = e2fgvi_checkpoint
        self.samcontroler = SamControler(
            self.sam_checkpoint, args.sam_model_type, args.device
        )
        self.xmem = BaseTracker(
            self.xmem_checkpoint, device=args.device, font_size=args.font_size
        )
        self.baseinpainter = BaseInpainter(self.e2fgvi_checkpoint, args.device)

    def first_frame_click(
        self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True
    ):
        mask, logit, painted_image = self.samcontroler.first_frame_click(
            image, points, labels, multimask
        )
        return mask, logit, painted_image

    def generator(
        self,
        images: list,
        template_mask: np.ndarray,
        video_description: str,
        objects_descriptions: list,
        language: str,
    ):
        masks = []
        logits = []
        painted_images = []
        objects_descriptions = "\n".join(objects_descriptions)
        self_play_conversation = generate_self_play_conversation(
            video_description, objects_descriptions, lan=language
        )
        formatted_conversation = format_generated_conversation(self_play_conversation)
        parallel_conversation = self.generate_list(images, formatted_conversation)
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i == 0:
                mask, logit, painted_image = self.xmem.track(
                    images[i], template_mask, cur_turn_dialogue=parallel_conversation[i]
                )
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)

            else:
                mask, logit, painted_image = self.xmem.track(
                    images[i], cur_turn_dialogue=parallel_conversation[i]
                )
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images

    def generate_list(self, first_list, second_list):
        result_list = [
            second_list[int(i // (len(first_list) / len(second_list)))]
            for i in range(len(first_list))
        ]
        return result_list


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument(
        "--port",
        type=int,
        default=6080,
        help="only useful when running gradio applications",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mask_save", default=False)
    parser.add_argument("--checkpoints_dir", default="./checkpoints")
    parser.add_argument("--font_size", default=30, type=int)
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args


if __name__ == "__main__":
    masks = None
    logits = None
    painted_images = None
    images = []
    image = np.array(PIL.Image.open("/hhd3/gaoshang/truck.jpg"))
    args = parse_augment()
    # images.append(np.ones((20,20,3)).astype('uint8'))
    # images.append(np.ones((20,20,3)).astype('uint8'))
    images.append(image)
    images.append(image)

    mask = np.zeros_like(image)[:, :, 0]
    mask[0, 0] = 1
    trackany = TrackingAnything(
        "/ssd1/gaomingqi/checkpoints/sam_vit_h_4b8939.pth",
        "/ssd1/gaomingqi/checkpoints/XMem-s012.pth",
        args,
    )
    masks, logits, painted_images = trackany.generator(images, mask)

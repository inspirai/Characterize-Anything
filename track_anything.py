import PIL
from tqdm import tqdm

from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
from inpainter.base_inpainter import BaseInpainter
import numpy as np
import argparse
import re

can_mouse_conversation = """
Coca-Cola Can: 嗨，你好！我是可乐罐，你是什么东西呢？
Computer Mouse: 嗨，我是电脑鼠标，你好！
Coca-Cola Can: 哦，原来是鼠标啊，我还以为是一只小老鼠呢！
Computer Mouse: 哈哈，我可不是小老鼠。不过，我们倒是有些共同点。
Coca-Cola Can: 哦？什么共同点？
Computer Mouse: 我们都是在办公室里工作的必备物品。你是为了让人们解渴，而我是为了让人们更方便地操作电脑。
Coca-Cola Can: 嗯，听起来很不错。不过，我觉得我更有趣一些，毕竟喝可乐比点鼠标更有乐趣嘛！
Computer Mouse: 哈哈，也许是吧。但是，你也不能太贪玩了，要是别人把你喝掉了，那就不好了。
Coca-Cola Can: 哎呀，你别吓唬我啊！我可是有灵魂的！
Computer Mouse: 哈哈，好了好了，我开玩笑的。不过，你还是要注意安全，毕竟我们都是为了人类服务的。
Coca-Cola Can: 好的好的，我会注意的。不过，你也要多注意保养啊，不然可就赚不到工资了！
Computer Mouse: 哈哈，你真逗！好的，我们都要好好干活，为人类的工作生活尽一份力！
Coca-Cola Can: 没错，干杯！
"""


def format_conversation(conversation):
    # 使用正则表达式提取说话人和内容
    pattern = r"([^\n:]+): ([^\n]+)"
    matches = re.findall(pattern, conversation)

    # 将提取到的说话人和内容转化为指定的输出格式
    dialogue_list = [(0 if m[0] == "Coca-Cola Can" else 1, m[1]) for m in matches]

    # 输出转化后的对话列表
    print(dialogue_list)
    return dialogue_list


class TrackingAnything:
    def __init__(self, sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.e2fgvi_checkpoint = e2fgvi_checkpoint
        self.samcontroler = SamControler(
            self.sam_checkpoint, args.sam_model_type, args.device
        )
        self.xmem = BaseTracker(self.xmem_checkpoint, device=args.device)
        self.baseinpainter = BaseInpainter(self.e2fgvi_checkpoint, args.device)

    # def inference_step(self, first_flag: bool, interact_flag: bool, image: np.ndarray,
    #                    same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
    #     if first_flag:
    #         mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
    #         return mask, logit, painted_image

    #     if interact_flag:
    #         mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
    #         return mask, logit, painted_image

    #     mask, logit, painted_image = self.xmem.track(image, logit)
    #     return mask, logit, painted_image

    def first_frame_click(
        self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True
    ):
        mask, logit, painted_image = self.samcontroler.first_frame_click(
            image, points, labels, multimask
        )
        return mask, logit, painted_image

    # def interact(self, image: np.ndarray, same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
    #     mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
    #     return mask, logit, painted_image

    def generator(self, images: list, template_mask: np.ndarray):
        masks = []
        logits = []
        painted_images = []

        formatted_conversation = format_conversation(can_mouse_conversation)
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

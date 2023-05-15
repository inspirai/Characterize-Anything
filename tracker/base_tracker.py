import os
import glob
import numpy as np

import torch
import yaml
import torch.nn.functional as F
from tracker.model.network import XMem
from inference.inference_core import InferenceCore
from tracker.util.mask_mapper import MaskMapper
from torchvision import transforms
from tracker.util.range_transform import im_normalization

from tools.painter import mask_painter
import progressbar
import cv2
from PIL import Image, ImageDraw, ImageFont


def add_text_box(image, bbox, img_border, text, font_size, fix_text_box, text_color):
    font_path = "resources/SourceHanSerifSC-Regular.otf"

    def wrap_text(text, font, fix_text_box):
        lines = []
        line = ""
        for char in text:
            line += char
            if font.getsize(line)[0] > fix_text_box[1]:
                lines.append(line[:-1])
                line = char
        lines.append(line)
        return lines

    def fit_text_in_box(
        img,
        text,
        position,
        textColor=(254, 234, 0),
        textSize=30,
        fix_text_box=None,
        img_border=None,
    ):
        if isinstance(img, np.ndarray):  # Check if the image is of OpenCV type
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        fontStyle = ImageFont.truetype(font_path, textSize, encoding="utf-8")
        lines = wrap_text(text, fontStyle, fix_text_box)

        while textSize > 1:
            total_text_height = len(lines) * fontStyle.getsize(lines[0])[1]
            if total_text_height <= fix_text_box[0]:
                break
            textSize -= 1
            fontStyle = ImageFont.truetype(font_path, textSize, encoding="utf-8")
            lines = wrap_text(text, fontStyle, fix_text_box)

        x_start, y_start = position[0], position[1] - (len(lines) + 1) * textSize

        if x_start < 0:
            x_start = 5
        elif x_start + fix_text_box[1] > img_border[1]:
            x_start = img_border[1] - fix_text_box[1]

        if y_start < 0:
            y_start = 5
        elif y_start + fix_text_box[0] > img_border[0]:
            y_start = img_border[0] - fix_text_box[0]

        y_offset = 0
        for line in lines:
            draw.text(
                (x_start, y_start + y_offset),
                line,
                textColor,
                font=fontStyle,
            )
            y_offset += fontStyle.getsize(line)[1]

        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    height, width, _ = img_border
    x1, y1, x2, y2 = bbox

    image = fit_text_in_box(
        image,
        text,
        (x1, y1),
        textSize=font_size,
        textColor=text_color,
        fix_text_box=fix_text_box,
        img_border=img_border,
    )

    return image


class BaseTracker:
    def __init__(
        self, xmem_checkpoint, device, sam_model=None, model_type=None
    ) -> None:
        """
        device: model device
        xmem_checkpoint: checkpoint of XMem model
        """
        # load configurations
        with open("tracker/config/config.yaml", "r") as stream:
            config = yaml.safe_load(stream)
        # initialise XMem
        network = XMem(config, xmem_checkpoint).to(device).eval()
        # initialise IncerenceCore
        self.tracker = InferenceCore(network, config)
        # data transformation
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                im_normalization,
            ]
        )
        self.device = device

        # changable properties
        self.mapper = MaskMapper()
        self.initialised = False

        # # SAM-based refinement
        # self.sam_model = sam_model
        # self.resizer = Resize([256, 256])

    @torch.no_grad()
    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(
            mask,
            (int(h / min_hw * self.size), int(w / min_hw * self.size)),
            mode="nearest",
        )

    @torch.no_grad()
    def track(
        self,
        frame,
        first_frame_annotation=None,
        cur_turn_dialogue=None,
        font_size=30,
        color1=0,
        color2=0,
        color3=0,
    ):
        """
        Input:
        frames: numpy arrays (H, W, 3)
        logit: numpy array (H, W), logit

        Output:
        mask: numpy arrays (H, W)
        logit: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """

        if first_frame_annotation is not None:  # first frame mask
            # initialisation
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask = None
            labels = None
        # prepare inputs
        frame_tensor = self.im_transform(frame).to(self.device)
        # track one frame
        probs, _ = self.tracker.step(frame_tensor, mask, labels)  # logits 2 (bg fg) H W
        # # refine
        # if first_frame_annotation is None:
        #     out_mask = self.sam_refinement(frame, logits[1], ti)

        # convert to mask
        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

        final_mask = np.zeros_like(out_mask)

        # map back
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        num_objs = final_mask.max()
        painted_image = frame
        bbox = None
        for obj in range(1, num_objs + 1):
            if np.max(final_mask == obj) == 0:
                continue
            # add text box
            if cur_turn_dialogue[0] == obj - 1:
                bbox = cv2.boundingRect((final_mask == obj).astype("uint8"))
            painted_image = mask_painter(
                painted_image, (final_mask == obj).astype("uint8"), mask_color=obj + 1
            )

        try:
            painted_image = add_text_box(
                painted_image,
                bbox,
                frame.shape,
                cur_turn_dialogue[1],
                font_size=font_size,
                text_color=(color1, color2, color3),
                fix_text_box=(3 * font_size, 10 * font_size),
            )
            # print(f'max memory allocated: {torch.cuda.max_memory_allocated()/(2**20)} MB')
        except:
            print("Error in adding text box")
            print(f"bbox:{bbox}")
            print(f"frame.shape:{frame.shape}")
            print(f"num_objs:{num_objs}")

        return final_mask, final_mask, painted_image

    def track_image_for_video(self, frame, all_masks, all_dialogue=None):
        print(
            f"frame:{frame}\nlen(all_masks):{len(all_masks)}\nlen(all_dialogue):{len(all_dialogue)}"
        )
        frame_array = np.array(frame)
        all_painted_image = []
        for _, cur_turn_dialogue in enumerate(all_dialogue):
            bbox = None
            for mask_idx in range(len(all_masks)):
                cur_mask = np.array(all_masks[mask_idx])
                print(f"cur_mask:{cur_mask}\n cur_mask_sum:{np.sum(cur_mask)}")
                # add text box
                if cur_turn_dialogue[0] == mask_idx - 1:
                    bbox = cv2.boundingRect((cur_mask == True).astype("uint8"))
                    print(bbox)

                    painted_image = add_text_box(
                        frame,
                        bbox,
                        frame_array.shape,
                        cur_turn_dialogue[1],
                        self.font_size,
                        fix_text_box=(3 * self.font_size, 10 * self.font_size),
                    )
                    all_painted_image.append(painted_image)
        all_painted_image_for_video = [
            pi for pi in all_painted_image for _ in range(60)
        ]
        return all_painted_image_for_video

    @staticmethod
    def get_image_border_pixels(image):
        height, width, _ = image.shape

        # Top border
        top_border = image[0, :, :]
        # Bottom border
        bottom_border = image[height - 1, :, :]
        # Left border
        left_border = image[:, 0, :]
        # Right border
        right_border = image[:, width - 1, :]

        return top_border, bottom_border, left_border, right_border

    @torch.no_grad()
    def sam_refinement(self, frame, logits, ti):
        """
        refine segmentation results with mask prompt
        """
        # convert to 1, 256, 256
        self.sam_model.set_image(frame)
        mode = "mask"
        logits = logits.unsqueeze(0)
        logits = self.resizer(logits).cpu().numpy()
        prompts = {"mask_input": logits}  # 1 256 256
        masks, scores, logits = self.sam_model.predict(
            prompts, mode, multimask=True
        )  # masks (n, h, w), scores (n,), logits (n, 256, 256)
        painted_image = mask_painter(
            frame, masks[np.argmax(scores)].astype("uint8"), mask_alpha=0.8
        )
        painted_image = Image.fromarray(painted_image)
        painted_image.save(f"/ssd1/gaomingqi/refine/{ti:05d}.png")
        self.sam_model.reset_image()

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()


##  how to use:
##  1/3) prepare device and xmem_checkpoint
#   device = 'cuda:2'
#   XMEM_checkpoint = '/ssd1/gaomingqi/checkpoints/XMem-s012.pth'
##  2/3) initialise Base Tracker
#   tracker = BaseTracker(XMEM_checkpoint, device, None, device)    # leave an interface for sam model (currently set None)
##  3/3)


if __name__ == "__main__":
    # video frames (take videos from DAVIS-2017 as examples)
    video_path_list = glob.glob(
        os.path.join(
            "/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/horsejump-high", "*.jpg"
        )
    )
    video_path_list.sort()
    # load frames
    frames = []
    for video_path in video_path_list:
        frames.append(np.array(Image.open(video_path).convert("RGB")))
    frames = np.stack(frames, 0)  # T, H, W, C
    # load first frame annotation
    first_frame_path = (
        "/ssd1/gaomingqi/datasets/davis/Annotations/480p/horsejump-high/00000.png"
    )
    first_frame_annotation = np.array(
        Image.open(first_frame_path).convert("P")
    )  # H, W, C

    # ------------------------------------------------------------------------------------
    # how to use
    # ------------------------------------------------------------------------------------
    # 1/4: set checkpoint and device
    device = "cuda:2"
    XMEM_checkpoint = "/ssd1/gaomingqi/checkpoints/XMem-s012.pth"
    # SAM_checkpoint= '/ssd1/gaomingqi/checkpoints/sam_vit_h_4b8939.pth'
    # model_type = 'vit_h'
    # ------------------------------------------------------------------------------------
    # 2/4: initialise inpainter
    tracker = BaseTracker(XMEM_checkpoint, device, None, device)
    # ------------------------------------------------------------------------------------
    # 3/4: for each frame, get tracking results by tracker.track(frame, first_frame_annotation)
    # frame: numpy array (H, W, C), first_frame_annotation: numpy array (H, W), leave it blank when tracking begins
    painted_frames = []
    for ti, frame in enumerate(frames):
        if ti == 0:
            mask, prob, painted_frame = tracker.track(frame, first_frame_annotation)
            # mask:
        else:
            mask, prob, painted_frame = tracker.track(frame)
        painted_frames.append(painted_frame)
    # ----------------------------------------------
    # 3/4: clear memory in XMEM for the next video
    tracker.clear_memory()
    # ----------------------------------------------
    # end
    # ----------------------------------------------
    print(f"max memory allocated: {torch.cuda.max_memory_allocated()/(2**20)} MB")
    # set saving path
    save_path = "/ssd1/gaomingqi/results/TAM/blackswan"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # save
    for painted_frame in progressbar.progressbar(painted_frames):
        painted_frame = Image.fromarray(painted_frame)
        painted_frame.save(f"{save_path}/{ti:05d}.png")

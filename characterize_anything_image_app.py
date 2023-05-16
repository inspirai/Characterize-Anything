import gdown
import cv2
import os
import sys
from PIL import Image

sys.path.append(sys.path[0] + "/tracker")
sys.path.append(sys.path[0] + "/tracker/model")
sys.path.append("MiniGPT-4")
from track_anything import TrackingAnything
from track_anything import parse_augment
import requests
import json
import torchvision
from tools.painter import mask_painter
import time
from typing import Union

import numpy as np
import torch
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
from role_play import make_completion


# ========================================
#             Model Initialization
# ========================================

print("Initializing Chat")
args = parse_augment()
cfg = Config(args)

model_config = cfg.model_cfg
gpu_id = int(args.device.split(":")[-1])
print(gpu_id)
model_config.device_8bit = gpu_id
print(f"args: {args}")
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(args.device)

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
    vis_processor_cfg
)
chat = Chat(model, vis_processor, device=args.device)
print("MiniGPT-4 Initialization Finished!")


# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath


def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print(
            "Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)"
        )
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath


# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "True",
    }
    return prompt


# extract frames from upload video
def get_frames_from_video(image_input, image_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    user_name = time.time()
    frame = np.array(image_input)
    operation_log = [
        ("", ""),
        (
            "Upload image already. Try click the image for adding targets to chat or self-play.",
            "Normal",
        ),
    ]
    # initialize video_state
    # todo:image name
    image_state = {
        "user_name": user_name,
        "image_name": "image1",
        "origin_image": frame,
        "painted_image": frame.copy(),
        "masks": [np.zeros(image_input.size, np.uint8)],
        "logits": [None],
    }
    image_info = "Image Name: {}, Image Size:{}".format(
        image_state["image_name"], image_input.size
    )

    # todo: image description
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(image_input, chat_state, img_list)
    print(f"llm_message: {llm_message}")
    user_message = "Describe this image."
    chat.ask(user_message, chat_state)
    image_description = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        temperature=1,
        max_new_tokens=300,
        max_length=2000,
    )[0]
    # image_description = "This image shows a can of soda and a computer mouse on a wooden desk. The can is made of metal with a white and red label. The mouse is a standard, wireless, rectangular shape with a white casing. There are no other objects or people in the image."
    print(f"image_description: {image_description}")
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(image_state["origin_image"])
    return (
        image_state,
        image_info,
        image_description,
        image_state["origin_image"],
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def run_example(example):
    return image_input


# get the select frame from gradio slider


# set the tracking end frame


def get_resize_ratio(resize_ratio_slider, interactive_state):
    interactive_state["resize_ratio"] = resize_ratio_slider

    return interactive_state


# use sam to get the mask
def sam_refine(
    image_state, point_prompt, click_state, interactive_state, evt: gr.SelectData
):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1

    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(image_state["origin_image"])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click(
        image=image_state["origin_image"],
        points=np.array(prompt["input_point"]),
        labels=np.array(prompt["input_label"]),
        multimask=prompt["multimask_output"],
    )
    image_state["masks"] = mask
    image_state["logits"] = logit
    image_state["painted_images"] = painted_image

    operation_log = [
        ("", ""),
        (
            "Use SAM for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Add mask button when you are satisfied with the segment",
            "Normal",
        ),
    ]
    return painted_image, image_state, interactive_state, operation_log


def add_multi_mask(image_state, interactive_state, mask_dropdown):
    try:
        mask = image_state["masks"]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append(
            "mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"]))
        )
        mask_dropdown.append(
            "mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"]))
        )
        select_frame, run_status = show_mask(
            image_state, interactive_state, mask_dropdown
        )

        operation_log = [
            ("", ""),
            (
                "Added a mask, use the mask select for target tracking or inpainting.",
                "Normal",
            ),
        ]
    except:
        operation_log = [
            ("Please click the left image to generate mask.", "Error"),
            ("", ""),
        ]
    return (
        interactive_state,
        gr.update(
            choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown
        ),
        select_frame,
        [[], []],
        operation_log,
    )


def add_mask_object_detection(image_state, interactive_state):
    seg_mask = image_state["masks"]
    mask_save_path = f"result/mask_{time.time()}.png"
    if not os.path.exists(os.path.dirname(mask_save_path)):
        os.makedirs(os.path.dirname(mask_save_path))
    seg_mask_img = Image.fromarray(seg_mask.astype("int") * 255.0)
    if seg_mask_img.mode != "RGB":
        seg_mask_img = seg_mask_img.convert("RGB")
    seg_mask_img.save(mask_save_path)
    print("seg_mask path: ", mask_save_path)
    print("seg_mask.shape: ", seg_mask.shape)

    image = image_state["origin_image"]
    if seg_mask is None:
        seg_mask = np.ones(image.size).astype(bool)

    image = load_image(image, return_type="pil")
    seg_mask = load_image(seg_mask, return_type="pil")

    seg_mask = seg_mask.resize(image.size)
    seg_mask = np.array(seg_mask) > 0

    box = new_seg_to_box(seg_mask)

    if np.array(box).size == 4:
        # [x0, y0, x1, y1], where (x0, y0), (x1, y1) represent top-left and bottom-right corners
        size = max(image.width, image.height)
        x1, y1, x2, y2 = box
        image_crop = np.array(image.crop((x1 * size, y1 * size, x2 * size, y2 * size)))
    elif np.array(box).size == 8:  # four corners of an irregular rectangle
        image_crop = cut_box(np.array(image), box)

    crop_save_path = f"result/crop_{time.time()}.png"
    image_crop_image = Image.fromarray(image_crop)
    image_crop_image.save(crop_save_path)
    print(f"croped image saved in {crop_save_path}")

    # todo: object description
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(image_crop_image, chat_state, img_list)
    user_message = "What’s the object name in this image?"
    chat.ask(user_message, chat_state)
    object_description = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        temperature=1,
        max_new_tokens=300,
        max_length=2000,
    )[0]
    # object_description = "this is a can"

    interactive_state["masks_object"].append(object_description)
    return image_state, interactive_state


def load_image(image: Union[np.ndarray, Image.Image, str], return_type="numpy"):
    """
    Load image from path or PIL.Image or numpy.ndarray to required format.
    """

    # Check if image is already in return_type
    if (
        isinstance(image, Image.Image)
        and return_type == "pil"
        or isinstance(image, np.ndarray)
        and return_type == "numpy"
    ):
        return image

    # PIL.Image as intermediate format
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if return_type == "pil":
        return image
    elif return_type == "numpy":
        return np.asarray(image)
    else:
        raise NotImplementedError()


def new_seg_to_box(seg_mask: Union[np.ndarray, Image.Image, str]):
    if type(seg_mask) == str:
        seg_mask = Image.open(seg_mask)
    elif type(seg_mask) == np.ndarray:
        seg_mask = Image.fromarray(seg_mask)
    seg_mask = np.array(seg_mask) > 0
    size = max(seg_mask.shape[0], seg_mask.shape[1])
    top, bottom = boundary(seg_mask)
    left, right = boundary(seg_mask.T)
    return [left / size, top / size, right / size, bottom / size]


def boundary(inputs):
    col = inputs.shape[1]
    inputs = inputs.reshape(-1)
    lens = len(inputs)

    start = np.argmax(inputs)
    end = lens - 1 - np.argmax(np.flip(inputs))

    top = start // col
    bottom = end // col

    return top, bottom


def cut_box(img, rect_points):
    w, h = get_w_h(rect_points)
    dst_pts = np.array(
        [
            [h, 0],
            [h, w],
            [0, w],
            [0, 0],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(rect_points.astype("float32"), dst_pts)
    cropped_img = cv2.warpPerspective(img, transform, (h, w))
    return cropped_img


def get_w_h(rect_points):
    w = np.linalg.norm(rect_points[0] - rect_points[1], ord=2).astype("int")
    h = np.linalg.norm(rect_points[0] - rect_points[3], ord=2).astype("int")
    return w, h


def clear_click(image_state, click_state):
    click_state = [[], []]
    template_frame = image_state["origin_image"]
    operation_log = [
        ("", ""),
        ("Clear points history and refresh the image.", "Normal"),
    ]
    return template_frame, click_state, operation_log


def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"] = []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("", ""), ("Remove all mask, please add new masks", "Normal")]
    return interactive_state, gr.update(choices=[], value=[]), operation_log


def show_mask(image_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = image_state["origin_image"]

    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(
            select_frame, mask.astype("uint8"), mask_color=mask_number + 2
        )

    operation_log = [
        ("", ""),
        ("Select {} for tracking or inpainting".format(mask_dropdown), "Normal"),
    ]
    return select_frame, operation_log


# inpaint


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.

    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    # height, width, layers = frames[0].shape
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # print(output_path)
    # for frame in frames:
    #     video.write(frame)

    # video.release()
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path


# check and download checkpoints if needed
SAM_checkpoint_dict = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
SAM_checkpoint_url_dict = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type]
sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type]
xmem_checkpoint = "XMem-s012.pth"
xmem_checkpoint_url = (
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
)
e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"
e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"

SAM_checkpoint = download_checkpoint(
    sam_checkpoint_url, args.checkpoints_dir, sam_checkpoint
)
xmem_checkpoint = download_checkpoint(
    xmem_checkpoint_url, args.checkpoints_dir, xmem_checkpoint
)
e2fgvi_checkpoint = download_checkpoint_from_google_drive(
    e2fgvi_checkpoint_id, args.checkpoints_dir, e2fgvi_checkpoint
)


# args.port = 12212
# args.device = "cuda:1"
# args.mask_save = True
def generate_prologue(object_description, language="English"):
    # todo: generate prologue
    prologue_prompt_zh = f"根据一段描述，生成一段描述中包含物体的开场白。\n例如：\n描述：The object in this image is a can of Coke.\n开场白：你好啊，我是一罐可乐. 根据以下描述生成开场白：\n描述: {object_description} \n开场白："
    prologue_prompt_en = f"Based on a given description, create a prologue that includes the object(s). \nexample:\ndescription: The object in this image is a can of Coke.\nprologue: Hi, I am a can of cocacola. \nGenerate prologue based on the following description:\ndescription: {object_description} \nprologue:"
    prompt = (
        prologue_prompt_zh
        if language == "zh" or language == "Chinese"
        else prologue_prompt_en
    )
    message = [
        {
            "role": "system",
            "content": prompt,
        }
    ]
    response = make_completion(message)
    # 根据mask number cache generated prologue
    print(
        f"{json.dumps({'message': message, 'response': response}, indent=2, ensure_ascii=False)}"
    )
    return response


def init_chatbot(
    interactive_state, mask_dropdown, language_dropdown, image_description
):
    # get chatbot prologue
    masks_object = interactive_state["masks_object"]

    active_masks = []
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        active_masks.append(masks_object[mask_number])

    prologue = ""
    if len(active_masks) == 0:
        init_msg = [(None, "Please select a mask first.")]
    elif len(active_masks) > 1:
        init_msg = [(None, "Please select only one mask.")]
    else:
        prologue = generate_prologue(active_masks[0], language_dropdown)
        init_msg = [(None, prologue)]

    # todo: hide chat with me button
    # show user input box
    prompt_zh = f"根据提供的图片描述、物体描述、物体开场白，接下来你要扮演这个物体，跟人类完成下面的对话，尽量保证对话自然、有趣，每次对话长度不要太长：\n\n图片描述: {image_description}\n物体描述: {masks_object[0]}\n"
    prompt_en = f"According to the provided image description, object description, and object prologue, you are going to play the role of this object and have a conversation with humans. Please try to make the conversation natural and interesting, and keep the length of each dialogue not too long.\n\nimage description: {image_description}\nobject description: {masks_object[0]}\n"
    message = [
        {
            "role": "system",
            "content": prompt_zh if language_dropdown == "Chinese" else prompt_en,
        },
        {"role": "assistant", "content": prologue},
    ]

    return init_msg, message


def get_img_self_chat(
    chatbot,
    image_state,
    template_frame,
    interactive_state,
    mask_dropdown,
    language_dropdown,
    image_description,
):
    objects_descriptions = []
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        objects_descriptions.append(
            f"{mask_number}. {interactive_state['masks_object'][mask_number]}"
        )
    all_masks = interactive_state["multi_mask"]["masks"]
    if len(objects_descriptions) < 0:
        chatbot = [(None, "Please select a mask first.")]

    all_painted_image_for_video = model.image_to_video_generator(
        template_frame,
        all_masks,
        image_description,
        objects_descriptions,
        language_dropdown,
    )

    # clear GPU memory
    model.xmem.clear_memory()
    video_output = generate_video_from_frames(
        all_painted_image_for_video,
        output_path="./result/track/{}".format(image_state["image_name"]),
        fps=30,
    )  # import video_input to name the output video
    return chatbot, video_output

    # todo: hide chat with me button
    # show user input box
    prompt_zh = f"根据提供的图片描述、物体描述、物体开场白，接下来你要扮演这个物体，跟人类完成下面的对话，尽量保证对话自然、有趣，每次对话长度不要太长：\n\n图片描述: {image_description}\n物体描述: {masks_object[0]}\n"
    prompt_en = f"According to the provided image description, object description, and object prologue, you are going to play the role of this object and have a conversation with humans. Please try to make the conversation natural and interesting, and keep the length of each dialogue not too long.\n\nimage description: {image_description}\nobject description: {masks_object[0]}\n"
    message = [
        {
            "role": "system",
            "content": prompt_zh if language == "Chinese" else prompt_en,
        },
        {"role": "assistant", "content": prologue},
    ]

    return init_msg, message


def user(user_message, history):
    return history + [[user_message, None]]


def predict(query, history):
    history.append({"role": "user", "content": query})
    # todo: chatgpt response
    # response = "this is a response"
    response = make_completion(history)
    history.append({"role": "assistant", "content": response})

    messages = [
        (history[i]["content"], history[i + 1]["content"])
        for i in range(0, len(history) - 1, 2)
    ]
    messages[0] = (None, messages[0][1])

    print(json.dumps(history, indent=4, ensure_ascii=False))
    return messages, history


# initialize sam, xmem, e2fgvi models
model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args)

title = """<p><h1 align="center">Characterize-Anything</h1></p>"""

with gr.Blocks() as iface:
    """
    state for
    """
    click_state = gr.State([[], []])
    interactive_state = gr.State(
        {
            "inference_times": 0,
            "negative_click_times": 0,
            "positive_click_times": 0,
            "mask_save": args.mask_save,
            "multi_mask": {"mask_names": [], "masks": []},
            "track_end_number": None,
            "resize_ratio": 1,
            "masks_object": [],
        }
    )

    image_state = gr.State(
        {
            "user_name": "",
            "image_name": "",
            "origin_image": None,
            "painted_image": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
        }
    )
    gr.Markdown(title)

    with gr.Row():
        # for user video input
        with gr.Column():
            image_input = gr.Image(
                type="pil", interactive=True, elem_id="image_upload"
            ).style(height=480)
            extract_frames_button = gr.Button(
                value="Get Image info", interactive=True, variant="primary"
            )
        with gr.Column():
            image_info = gr.Textbox(label="Image Info")
            image_description = gr.Textbox(label="Image Description")
            resize_ratio_slider = gr.Slider(
                minimum=0.02,
                maximum=1,
                step=0.02,
                value=1,
                label="Resize ratio",
                visible=True,
            )

    gr.Markdown("""<p></p><p><h1 align="center">Chat</h1></p><p></p>""")
    with gr.Row():
        # put the template frame under the radio button
        with gr.Column():
            # click points settins, negative or positive, mode continuous or single
            with gr.Row():
                point_prompt = gr.Radio(
                    choices=["Positive", "Negative"],
                    value="Positive",
                    label="Point Prompt",
                    interactive=True,
                    visible=False,
                )
                remove_mask_button = gr.Button(
                    value="Remove mask", interactive=True, visible=False
                )
                clear_button_click = gr.Button(
                    value="Clear Clicks", interactive=True, visible=False
                ).style(height=160)
            with gr.Row():
                Add_mask_button = gr.Button(
                    value="Add mask", interactive=True, visible=False
                )
            template_frame = gr.Image(
                type="pil",
                interactive=True,
                elem_id="template_frame",
                visible=False,
            ).style(height=360)

        with gr.Column():
            run_status = gr.HighlightedText(
                value=[
                    ("Text", "Error"),
                    ("to be", "Label 2"),
                    ("highlighted", "Label 3"),
                ],
                visible=False,
            )
            lang_dropdown = gr.Dropdown(
                choices=["English", "Chinese"],
                value="English",
                label="Language",
            )
            mask_dropdown = gr.Dropdown(
                multiselect=True,
                value=[],
                label="Mask selection",
                info=".",
                visible=False,
            )
            mask_description = gr.Dropdown(
                multiselect=True,
                value=[],
                label="Mask description",
                info=".",
                visible=False,
            )

            chat_button = gr.Button(
                value="Chat with me (one object)", interactive=True, visible=True
            )

            self_chat_button = gr.Button(
                value="Objects self chat (multi objects)",
                interactive=True,
                visible=True,
            )

            chatbot = gr.Chatbot(label="Chatbox").style(height=550, scale=0.5)
            video_output = gr.Video(autosize=True, visible=False).style(height=360)

            msg = gr.Textbox(
                placeholder="Enter text and press enter",
            ).style(container=False)

            message_state = gr.State()
    # first step: get the video information
    extract_frames_button.click(
        fn=get_frames_from_video,
        inputs=[image_input, image_state],
        outputs=[
            image_state,
            image_info,
            image_description,
            template_frame,
            point_prompt,
            clear_button_click,
            Add_mask_button,
            template_frame,
            mask_dropdown,
            remove_mask_button,
        ],
    )

    resize_ratio_slider.release(
        fn=get_resize_ratio,
        inputs=[resize_ratio_slider, interactive_state],
        outputs=[interactive_state],
        api_name="resize_ratio",
    )

    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[image_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, image_state, interactive_state],
    )

    # add different mask
    Add_mask_button.click(
        fn=add_multi_mask,
        inputs=[image_state, interactive_state, mask_dropdown],
        outputs=[
            interactive_state,
            mask_dropdown,
            template_frame,
            click_state,
            run_status,
        ],
    ).then(
        fn=add_mask_object_detection,
        inputs=[image_state, interactive_state],
        outputs=[image_state, interactive_state],
    )

    chat_button.click(
        fn=init_chatbot,
        inputs=[interactive_state, mask_dropdown, lang_dropdown, image_description],
        outputs=[chatbot, message_state],
    )

    self_chat_button.click(
        fn=lambda x, y: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[chatbot, video_output],
        outputs=[chatbot, video_output],
    ).then(
        fn=get_img_self_chat,
        inputs=[
            chatbot,
            image_state,
            template_frame,
            interactive_state,
            mask_dropdown,
            lang_dropdown,
            image_description,
        ],
        outputs=[chatbot, message_state],
    )

    msg.submit(user, [msg, chatbot], chatbot, queue=False).then(
        predict, [msg, message_state], [chatbot, message_state]
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, run_status],
    )

    # click to get mask
    mask_dropdown.change(
        fn=show_mask,
        inputs=[image_state, interactive_state, mask_dropdown],
        outputs=[template_frame, run_status],
    )

    # clear input
    image_input.clear(
        lambda: (
            {
                "user_name": "",
                "image_name": "",
                "origin_image": None,
                "painted_image": None,
                "masks": None,
                "inpaint_masks": None,
                "logits": None,
            },
            {
                "inference_times": 0,
                "negative_click_times": 0,
                "positive_click_times": 0,
                "mask_save": args.mask_save,
                "multi_mask": {"mask_names": [], "masks": []},
                "track_end_number": 0,
                "resize_ratio": 1,
            },
            [[], []],
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False, value=[]),
            gr.update(visible=False),
            gr.update(visible=False),
        ),
        [],
        [
            image_state,
            interactive_state,
            click_state,
            template_frame,
            point_prompt,
            clear_button_click,
            Add_mask_button,
            template_frame,
            mask_dropdown,
            remove_mask_button,
            run_status,
        ],
        queue=False,
        show_progress=False,
    )

    # points clear
    clear_button_click.click(
        fn=clear_click,
        inputs=[
            image_state,
            click_state,
        ],
        outputs=[template_frame, click_state, run_status],
    )
    # set example
    gr.Markdown("##  Examples")
    gr.Examples(
        examples=[
            os.path.join(os.path.dirname(__file__), "./test_sample/", test_sample)
            for test_sample in [
                "test-sample1.mp4",
            ]
        ],
        fn=run_example,
        inputs=[image_input],
        outputs=[image_input],
        # cache_examples=True,
    )
iface.queue(concurrency_count=1)
iface.launch(
    debug=True, enable_queue=True, server_port=args.port, server_name="0.0.0.0"
)
# iface.launch(debug=True, enable_queue=True)

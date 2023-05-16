import json

import openai


openai.api_key = ""


def generate_self_play_conversation(video_description, objects_descriptions, lan="zh"):
    prompt_template_zh = """作为一个富有想象力的编剧，你的任务是根据一段场景描述，以及其中的物体描述，创造出一个有趣的对话故事。这些物体以第一人称来进行对话，它们有生命和独特的个性。你可以自由发挥，但要确保对话灵活、生动、逼真。

示例：
场景描述：
This image shows a woman in green yoga pants standing next to a small brown and white dog sitting on a blue yoga mat. The woman is wearing a watch and has a small smile on her face. The dog is looking up at the woman with a curious expression. There is a white wall in the background and some green plants in a pot in the foreground. The overall atmosphere of the image is peaceful and relaxed. The lighting is bright and natural. The composition of the image is balanced, with the woman and dog positioned off-center. The image is framed by the blue mat and the wall in the background.
物体描述：
0. The object in this image is a woman.
1. The object in this image is a dog.

生成对话：
女人(0)：嗨，小狗，你今天怎么样？
狗(1)：汪汪！我感觉很好，谢谢你。你看起来也很开心，是做瑜伽让你感觉如此放松吗？
女人(0)：是的，小狗，瑜伽真的很有益，可以帮助我们放松身心，让我们更健康。
狗(1)：我也想做瑜伽！能教我吗？
女人(0)：当然可以，小狗！我们可以一起练习瑜伽，带给你一份健康和快乐。
狗(1)：太棒了！我可以和你一起练习瑜伽，变得更加灵活和强壮。
女人(0)：对啊，小狗，我们可以一起变得更加健康和快乐。你是我最好的瑜伽伙伴。
狗(1)：汪汪！我也很开心能和你一起做瑜伽，女人。我们一起变得更加强大和平静！

以下是场景的描述：
{video_description}
物体描述：
{objects_descriptions}

根据以上场景和物体的描述生成一段物体之间的中文对话：""".format(
        video_description=video_description, objects_descriptions=objects_descriptions
    )

    prompt_template_en = """As an imaginative screenwriter, your task is to create an interesting dialogue story based on a scene description and the descriptions of objects within it. These objects engage in conversation in the first person, and they have life and unique personalities. You can freely play around with the story, but make sure the dialogue is flexible, vivid, and realistic.

Example:
Scene description:
This image shows a woman in green yoga pants standing next to a small brown and white dog sitting on a blue yoga mat. The woman is wearing a watch and has a small smile on her face. The dog is looking up at the woman with a curious expression. There is a white wall in the background and some green plants in a pot in the foreground. The overall atmosphere of the image is peaceful and relaxed. The lighting is bright and natural. The composition of the image is balanced, with the woman and dog positioned off-center. The image is framed by the blue mat and the wall in the background.
Object description:
0. The object in this image is a woman.
1. The object in this image is a dog.

Generated dialogue:
Dog(0): What are we doing here again?
Woman(1): We're doing yoga, buddy.
Dog(0): Yoga? Is that like fetch?
Woman(1): No, not quite. It's more about stretching and breathing.
Dog(0): I don't know about all that. I prefer chasing squirrels and napping.
Woman(1): Trust me, it feels good. And it's good for you too.
Dog(0): Okay, I'll give it a try. But I might need your help with some of the poses.
Woman(1): I'm happy to help. Let's start with downward dog.
Dog(0): Of course, down dog. Can't escape that one.

Here is the scene description:
{video_description}
Object description:
{objects_descriptions}

Generate a dialogue between the objects based on the scene and object descriptions above:""".format(
        video_description=video_description, objects_descriptions=objects_descriptions
    )

    message = [
        {
            "role": "system",
            "content": prompt_template_zh
            if lan == "zh" or lan == "Chinese"
            else prompt_template_en,
        }
    ]
    print(f"message: {json.dumps(message, indent=4, ensure_ascii=False)}")
    try:
        return make_completion(message)
        # return fake_response
    except Exception as e:
        print(e)
        return ""


def make_completion(history):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=history,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    response = completion["choices"][0]["message"]["content"]

    return response


if __name__ == "__main__":
    video_description = "This image shows a white mouse sitting on a wooden table next to a can of soda. The mouse has red eyes and is made of plastic. The can of soda is red with white lettering and has a tab on the top. The table is plain with no other objects on it. There is a wooden chair in the background. The overall color scheme of the image is red and white."
    objects_descriptions = "3. The object in this image is a can of Coke.\n2. The object in this image is a mouse."
    print(
        generate_self_play_conversation(
            video_description, objects_descriptions, lan="zh"
        )
    )
    print(
        generate_self_play_conversation(
            video_description, objects_descriptions, lan="en"
        )
    )

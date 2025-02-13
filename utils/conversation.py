import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<graph>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Graph><graph></Graph>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<graph>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""

        elif self.sep_style == SeparatorStyle.LLAMA_3:#TODO
            ret = "<|begin_of_text|>"
            if self.system:
                ret += self.system
            else:
                ret += ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    ret += f"{message.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])
    #
    # def get_images(self, return_pil=False):
    #     images = []
    #     for i, (role, msg) in enumerate(self.messages[self.offset:]):
    #         if i % 2 == 0:
    #             if type(msg) is tuple:
    #                 import base64
    #                 from io import BytesIO
    #                 from PIL import Image
    #                 msg, image, image_process_mode = msg
    #                 if image_process_mode == "Pad":
    #                     def expand2square(pil_img, background_color=(122, 116, 104)):
    #                         width, height = pil_img.size
    #                         if width == height:
    #                             return pil_img
    #                         elif width > height:
    #                             result = Image.new(pil_img.mode, (width, width), background_color)
    #                             result.paste(pil_img, (0, (width - height) // 2))
    #                             return result
    #                         else:
    #                             result = Image.new(pil_img.mode, (height, height), background_color)
    #                             result.paste(pil_img, ((height - width) // 2, 0))
    #                             return result
    #                     image = expand2square(image)
    #                 elif image_process_mode in ["Default", "Crop"]:
    #                     pass
    #                 elif image_process_mode == "Resize":
    #                     image = image.resize((336, 336))
    #                 else:
    #                     raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
    #                 max_hw, min_hw = max(image.size), min(image.size)
    #                 aspect_ratio = max_hw / min_hw
    #                 max_len, min_len = 800, 400
    #                 shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
    #                 longest_edge = int(shortest_edge * aspect_ratio)
    #                 W, H = image.size
    #                 if longest_edge != max(image.size):
    #                     if H > W:
    #                         H, W = longest_edge, shortest_edge
    #                     else:
    #                         H, W = shortest_edge, longest_edge
    #                     image = image.resize((W, H))
    #                 if return_pil:
    #                     images.append(image)
    #                 else:
    #                     buffered = BytesIO()
    #                     image.save(buffered, format="PNG")
    #                     img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    #                     images.append(img_b64_str)
    #     return images

    # def to_gradio_chatbot(self):
    #     ret = []
    #     for i, (role, msg) in enumerate(self.messages[self.offset:]):
    #         if i % 2 == 0:
    #             if type(msg) is tuple:
    #                 import base64
    #                 from io import BytesIO
    #                 msg, image, image_process_mode = msg
    #                 max_hw, min_hw = max(image.size), min(image.size)
    #                 aspect_ratio = max_hw / min_hw
    #                 max_len, min_len = 800, 400
    #                 shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
    #                 longest_edge = int(shortest_edge * aspect_ratio)
    #                 W, H = image.size
    #                 if H > W:
    #                     H, W = longest_edge, shortest_edge
    #                 else:
    #                     H, W = shortest_edge, longest_edge
    #                 image = image.resize((W, H))
    #                 buffered = BytesIO()
    #                 image.save(buffered, format="JPEG")
    #                 img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    #                 img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
    #                 msg = img_str + msg.replace('<image>', '').strip()
    #                 ret.append([msg, None])
    #             else:
    #                 ret.append([msg, None])
    #         else:
    #             ret[-1][-1] = msg
    #     return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        # if len(self.get_images()) > 0:
        #     return {
        #         "system": self.system,
        #         "roles": self.roles,
        #         "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
        #         "offset": self.offset,
        #         "sep": self.sep,
        #         "sep2": self.sep2,
        #     }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
# TODO修改system
conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


conv_childtext = Conversation(
    system="你是一名专业的儿童文学评委，具备多年评估儿童作文的经验。 "
           "你能够理解用户提供的图内容。 ",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_edge_pred = Conversation(
    system="你是一名儿童文学专家，能够理解儿童的叙事中提到的事件与事件关系。",
    roles=("USER", "ASSISTANT"),
    version="no_graph",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


conv_childtext_nograph = Conversation(
    system="你是一名专业的儿童文学评委，具备多年评估儿童作文的经验。 "
           "现在的任务是对儿童叙事作文进行评估。",
    roles=("USER", "ASSISTANT"),
    version="no_graph",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_childtext_llama2 = Conversation(
    system="你是一名专业的儿童文学评委，具备多年评估儿童作文的经验。"
           "你能够理解用户提供的图内容。 ",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

# llama3 template
# reference: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
# reference: https://github.com/meta-llama/llama3/blob/0cee08ec68f4cfc0c89fe4a9366d82679aaa2a66/llama/tokenizer.py#L222
conv_childtext_llama_3 = Conversation(
        system="<|start_header_id|>system<|end_header_id|>\n\n你是一名专业的儿童文学评委，具备多年评估儿童作文的经验。你能够看懂叙事图的内容。<|eot_id|>",
        roles=("user", "assistant"),
        version="llama_v3",
        sep_style=SeparatorStyle.LLAMA_3,
        messages=(),
        sep="<|end_header_id|>\n",
        sep2 = "<|eot_id|>",
        offset=0
        #stop_str="<|eot_id|>",
        #stop_token_ids=[128001, 128009],#128001=<|end_of_text|>;128009=<|eot_id|>
)

conv_llaga_llama_2 = Conversation(
    system="You are a helpful language and graph assistant. "
           "You are able to understand the graph content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)




conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)



# conv_mpt = Conversation(
#     system="""<|im_start|>system
# A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
#     roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
#     version="mpt",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.MPT,
#     sep="<|im_end|>",
# )

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="</s>",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the graph content that the user provides, and assist the user with a variety of tasks using natural language."
           "The graph content will be provided with the following format: <Graph>graph content</Graph>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the graph content that the user provides, and assist the user with a variety of tasks using natural language."
           "The graph content will be provided with the following format: <Graph>graph content</Graph>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)
# A template with a one-shot conversation example

conv_one_shot=    Conversation(
        system ="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Got any creative ideas for a 10 year old’s birthday?",
            ),
            (
                "Assistant",
                """Of course! Here are some creative ideas for a 10-year-old's birthday party:
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!""",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="\n### ",
        sep2="###",
        # stop_str="###",#只在推理时用到
    )


# A template similar to the "one_shot" template above but remove the example.

conv_zero_shot=    Conversation(
        # name="zero_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="\n### ",
        sep2="###",
        # stop_str="###",
    )


default_conversation = conv_childtext
conv_templates = {
    "default": conv_childtext,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "llaga_llama_2": conv_llaga_llama_2,
    "mpt": conv_mpt,
    "conv_childtext":conv_childtext,
    "conv_childtext_nograph":conv_childtext_nograph,
    "conv_t5":conv_zero_shot,
    "conv_childtext_llama2":conv_childtext_llama2,
    "conv_edge_pred":conv_edge_pred
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())

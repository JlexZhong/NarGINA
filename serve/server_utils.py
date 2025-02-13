


import pydot
import plotly.graph_objects as go
from PIL import Image
from paddlespeech.cli.asr.infer import ASRExecutor
import requests

import re
import sys
import os
import requests
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append("./")
sys.path.append("./utils")
sys.path.append("./train")
import torch
from PIL import Image
from encode_graph import SentenceEncoder
import os
from contrastive_learning.GRACE import GAT, Encoder
from utils.utils import get_model_name_from_path
import torch
torch.autograd.set_detect_anomaly(True)
import os
import json
from sentence_transformers import SentenceTransformer
import GCL.augmentors as A
from model.builder import load_pretrained_model
from utils.utils import get_model_name_from_path
from model.builder import load_pretrained_model

# parser = argparse.ArgumentParser()
# parser.add_argument("--model_base", type=str, default="/disk/NarGINA/weights/vicuna-7b-v1.5")
# parser.add_argument("--pretrained_embedding_type", type=str, default="GRACE_512")
# parser.add_argument("--conv_mode", type=str, default="conv_childtext")
# parser.add_argument("--temperature", type=float, default=0.8)
# parser.add_argument("--top_p", type=float, default=None)
# parser.add_argument("--num_beams", type=int, default=1)
# parser.add_argument("--prompt", type=str, default=None)
# parser.add_argument("--start", type=int, default=-1)
# parser.add_argument("--end", type=int, default=-1)
# parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
# parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
# parser.add_argument("--is_only_graph", type=bool, default=False)#TODO
# parser.add_argument("--is_trait_comment", type=bool, default=False)#TODO
# parser.add_argument("--is_trait", type=bool, default=True)#TODO
# parser.add_argument("--is_precompute_micro_metric", type=bool, default=False)#TODO
# parser.add_argument("--is_edge_str", type=bool, default=False)#TODO
# args = parser.parse_args()


EXAMPLE_ESSAY = """从前有一个小男孩，他养了一只青蛙。有一天晚上，小男孩睡觉的时候，青蛙偷偷地爬了出来。第二天早上小男孩醒来时，发现青蛙不见了。他急忙起来寻找，带着自己的小狗，开始四处寻找青蛙的踪影。
小男孩和小狗来到窗户旁边，透过窗子寻找青蛙的身影。小狗套上了一个瓶子，小男孩穿着靴子，准备仔细地搜寻每一个角落。突然，小狗跳了下去，小男孩立刻冲过去抱住小狗，继续在森林旁边呼喊着小青蛙的名字。可是，周围一片寂静，完全没有小青蛙的声音。
在森林里，小男孩发现了一个洞，洞旁边还有一棵大树，树上有一个鸟窝，里面似乎藏着什么东西。他往洞里喊道：“小青蛙，你在这里吗？”突然，一只小老鼠从洞里跑出来，回答说：“我不是小青蛙，我是小老鼠。”就在这时，小狗看到了树上鸟窝里的蜜蜂飞了出来，吓得目瞪口呆。小狗不小心弄翻了东西，惹得许多蜜蜂都飞了出来。
继续寻找的过程中，小男孩发现了一棵大树，他猜测树洞里也许会藏着小青蛙，于是再次喊道：“小青蛙，你在这里吗？”然而，从树洞里飞出来的却是一只猫头鹰，它对小男孩说：“附近没有小青蛙，我是猫头鹰！”猫头鹰追着小男孩，重复着没有小青蛙的消息。
小男孩一边喊着“小青蛙，你在哪里？”，一边在森林中四处寻找，结果不小心跌到了树干上。就在他疑惑之际，发现旁边有一只鹿。鹿在这片森林中游荡，看到小男孩便将他背在了背上，帮他继续寻找。然而，走到河边时，鹿一不小心把小男孩和小狗摔了下去，他们跌入水中。
小男孩和小狗在河中游了一会儿，但没有发现任何踪迹。鹿低头喝水时，也没注意到这边是个悬崖，无法继续靠近水源。小男孩发现了一根树枝，便爬了上去，在上面稍作休息。这时，他隐约听到“呱呱呱”的声音，急忙叫小狗说：“好像听到青蛙的声音了！”
他们翻过一片草丛，终于看到了小青蛙，原来青蛙和它的家人都在这里，几只小青蛙和两只大青蛙正快乐地叫着。小男孩和小狗带着他们的小青蛙，终于高高兴兴地回到了家。"""

EXAMPLE_ESSAY_v2 = """有一天，有一个小男孩养了一只青蛙。这个小男孩还养了一只狗，这个狗对这个小青蛙也很好奇。当天晚上睡觉的时候，这只青蛙偷偷地从瓶子里钻了出来。第二天早上，小男孩和他的狗都发现青蛙不见了。他们就去找这青蛙到底去哪儿了。先翻遍了小男孩的靴子，发现没有。小狗钻进了瓶子里去找，结果一不小心被这个瓶子卡住了。
他们又去窗台上，去呼喊。一不小心，这只小狗就从窗台上掉了下来。小男孩下去，接住了这只小狗。于是，他们又走进森林里，去寻找。首先，看到了一个马蜂窝。树的旁边还有一个洞，小男孩朝这个洞里去呼叫，没有发现。而这只小狗去摇晃了有马蜂窝的树，这个马蜂窝就掉了下来，蜜蜂都飞了出来。
接着，小男孩又爬到树洞里去找。树洞里没有青蛙，是一只老鹰。老鹰飞了出来，把小男孩撞倒在地。然后他们俩又去寻找，走到一个山坡上。还是没有找到青蛙，而是一只麋鹿。这只麋鹿，骑着小男孩。小狗在旁边追逐着，把他们赶到了山脚下。
山脚下正好有一片池塘，小男孩和小狗都掉进了池塘里。岸边有一棵树。小男孩让小狗不要出声，他往树后面一看，有两只青蛙。接着，又来了几只青蛙，看来是一家人。于是，小男孩在池塘里，找到了青蛙。"""

COMMENT_GENERATE_PROMPT = f"""
### 满分参考范文：
从前有一个男孩，他有一只狗和一只宠物青蛙。 他把青蛙放在他卧室的一个大罐子里。一天晚上，当他和他的狗在睡觉时，青蛙从罐子里爬了出来。 它从一扇打开的窗口跳了出去。第二天早上，当男孩和狗醒来时，他们发现罐子是空的。小男孩到处找青蛙。小狗也找青蛙。当狗试图往罐子里看时，它的头被卡住了。男孩从开着的窗户喊道：“青蛙，你在哪里？”。狗探出窗外，罐子仍然卡在它的头上。罐子太重了，小狗头朝下从窗户掉了下去。罐子摔碎了。小男孩跳下窗台很生气的抱着小狗 男孩和小狗在外面找青蛙。男孩呼喊着青蛙。当小狗对着蜂巢里的蜜蜂叫的时候，小男孩朝地洞里喊。一只地鼠从洞里钻出来，正好咬在男孩的鼻子上。与此同时，小狗还在骚扰蜜蜂，跳到树上对它们叫。蜂窝掉下来，所有的蜜蜂都飞出来了。小男孩一点儿都没有注意到小狗。他注意到树上有一个大洞。于是他爬上树，朝洞里喊。突然，一只猫头鹰从洞里猛扑出来，把男孩撞倒在地。小狗以最快的速度从男孩身边跑过，因为蜜蜂在追它。猫头鹰一路追着小男孩，追到一块大石头前。小男孩爬上那块石头，再次呼喊着青蛙。他抓住一些树枝，这样他就不会掉下去。但是树枝并不是真的树枝，而是鹿角。鹿把小男孩放在头上。鹿开始奔跑，小男孩还在它头上。小狗也跟着跑。它们越来越接近悬崖。鹿突然停了下来，男孩和狗从悬崖边掉了下去。悬崖下面有一个池塘。他们一个接一个地扑通落进了池塘。他们听到了熟悉的声音。男孩让小狗安静一点。他们悄悄爬上去，朝一根大原木后面看去。他们翻过原木，看见两只大青蛙。还有一些小青蛙，其中一只跳向男孩。小男孩把一只小青蛙带回家，并开心地和其他青蛙说再见。
### 输出格式
1.宏观结构维度评语
    - 相比于满分参考范文，缺失的关键事件有：
    - 不合理的因果关系有：
2.微观结构维度评语
3.叙事心理描写维度评语

Question: 根据上述参考范文和儿童叙事文，从三个维度上给出评语，按照给定的输出格式输出你的回答。"""

CRITERIA = """### 评分标准
共三个维度，分数0-10。
1. **宏观结构维度**
    - 故事是否有明确的开端、发展、高潮和结尾？
    - 故事的整体结构是否连贯，有无突兀的跳跃或不合理的情节？
    - 角色的行动是否有合理的因果关系？
2. **微观结构维度**
    - 词汇生成是否丰富？（参考词汇总数和不同词汇出现率）
    - 句子结构是否复杂？（参考平均句长和句法复杂度）
    - 是否使用了修辞手法？
3. **叙事心理描写维度**
    - 角色的情感表达是否与情节发展一致？
    - 是否有角色的心理描写？（对于儿童而言，反映基本情绪反应即可）"""

# 初始化ASR识别器
asr = ASRExecutor()

dtype = torch.bfloat16

# 定义加载模型的函数
def load_model(model_path, model_base):
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)    
    tokenizer, model, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        cache_dir="../../checkpoint"  # 你可以根据需要指定缓存路径
    )
    model = model.to("cuda").to(dtype)
    return tokenizer, model

def get_graph_encoder(encoder_weight_path,model_path):

    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    #encoder_weight_path = '/disk/NarGINA/contrastive_learning/weights/encoder_model_vicuna-7b-v1.5_hidden=512_numlayers=4_headnum=8_loss=4.32_weights.pth'
    # 使用正则表达式匹配参数及其值
    parameters = {
        "hidden": re.search(r"hidden=(\d+)", encoder_weight_path).group(1),
        "numlayers": re.search(r"numlayers=(\d+)", encoder_weight_path).group(1),
        "headnum": re.search(r"headnum=(\d+)", encoder_weight_path).group(1),
        "loss": re.search(r"loss=([\d.]+)", encoder_weight_path).group(1)
    }
    heads_num = int(parameters["headnum"]) 
    num_layers = int(parameters["numlayers"]) 
    hidden_dim = int(parameters["hidden"]) 
    proj_dim = int(parameters["hidden"]) 
    if "GRACE_512_STv2" in model_path:
        text_dim = 512
    elif "GRACE_512_ST" in model_path or "GRACE_512_STv1" in model_path:
        text_dim = 768
    elif "GRACE_512_7b" in model_path:
        text_dim = 4096
    gat = GAT(input_dim=text_dim, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=num_layers,heads=heads_num).to('cuda')
    graph_encoder_model = Encoder(encoder=gat, augmentor=(aug1, aug2), hidden_dim=hidden_dim, proj_dim = proj_dim).to('cuda')
    graph_encoder_model.load_state_dict(torch.load(encoder_weight_path))
    print("Graph Encoder model weights loaded.")
    return graph_encoder_model

def get_text_encoder(model_path):
    llm_b_size = 32
    if "GRACE_512_STv2" in model_path:
        return SentenceTransformer('/disk/NarGINA/weights/sentence-transformers')
    elif "GRACE_512_ST" in model_path: 
        return SentenceEncoder("ST", batch_size=llm_b_size,pooling=True)
    elif "GRACE_512_7b" in model_path:
        return SentenceEncoder("vicuna-7b-v1.5", batch_size=llm_b_size,pooling=True)


# 语音识别函数
def recognize_audio(audio):
    result = asr(audio_file=audio,force_yes=True)
    return result

def ST_emb_text(texts,encoder):
    text_emb = encoder.encode(texts,convert_to_tensor=True,show_progress_bar=True)
    return text_emb

def narrative_graph_api(input_text):
    # 分句（通过标点符号分句，考虑到中英文符号）
    sentences = re.split(r'[。！？!?\.]', input_text)
    sentences = [sent.strip() for sent in sentences if sent.strip()]  # 去除空白句子
    data = {
    'story':sentences
    }
    # 发起 POST 请求
    url = 'http://localhost:5000/narrative'
    response = requests.post(url, json=data)

    # 检查响应状态
    if response.status_code == 200:
        # 如果请求成功，处理返回的数据
        content = response.json()  # 获取返回的 JSON 数据
        return content
        print("Response:", content)
    else:
        print("Error:", response.json())
        return None
    

def post_graph_image():
    graph_img_path = '/disk/NarGINA/serve/asset/graph_v1.png'
    return Image.open(graph_img_path)
    url = "http://127.0.0.1:5000/get_image"

    response = requests.get(url)
    
    if response.status_code == 200:
        with open(graph_img_path, "wb") as f:
            f.write(response.content)
        print("Image downloaded successfully as downloaded_image.png")
        return Image.open(graph_img_path)
    else:
        print(f"Failed to get image. Status code: {response.status_code}")
        print(response.json())
        return None
    

def cl_model_encode(g,graph_encoder_model):
    graph_encoder_model.eval()
    with torch.no_grad():
        g.to("cuda")
        z, _, _ = graph_encoder_model(g.x, g.edge_index, g.edge_attr)
        g.x = z
        return g
    

def get_pydot_graph_img(content,output_file_name):
    label_dict = {"动机-因果": 0, "心理-因果": 1, "物理-因果": 2, "使能-因果": 3, "并列": 4, "无": 5}
    relation_colors = {
        '动机-因果': 'red',
        '心理-因果': 'blue',
        '物理-因果': 'green',
        "使能-因果": 'orange',
        "并列": 'purple',
        '无': 'black'
    }
    
    for count, story in enumerate(content):
        events = []
        for edge in story:
            if edge['first_event'] not in events:
                events.append(edge['first_event'])
            if edge['second_event'] not in events:
                events.append(edge['second_event'])

        events_dict = {event: idx for idx, event in enumerate(events)}

        # 创建主图（包含有边的节点）
        main_graph = pydot.Dot(graph_type='digraph', encoding='utf-8',prog='neato')
        main_graph.set_edge_defaults(fontname='SimHei')
        main_graph.set_node_defaults(fontname='SimHei')
        main_graph.set_node_defaults(style="filled", fillcolor="lightblue", shape="ellipse")

        # 设置垂直布局
        main_graph.set('rankdir', 'LR')

        # 创建孤立节点图
        isolated_graph = pydot.Dot(graph_type='digraph', encoding='utf-8')
        isolated_graph.set_node_defaults(fontname='SimHei')
        isolated_graph.set('rankdir', 'TB')

        # 添加边
        for edge in story:
            first_event_id = events_dict[edge['first_event']]
            second_event_id = events_dict[edge['second_event']]
            relation = edge['relation']
            color = relation_colors[relation]

            if relation != '无':
                main_graph.add_edge(pydot.Edge(f"{first_event_id}\n{edge['first_event']}", 
                                               f"{second_event_id}\n{edge['second_event']}", 
                                               label=relation, color=color))
        # 找出所有孤立节点并添加到孤立图中
        all_connected_nodes = set()  # 用于存储所有在有效边中出现的节点

        # 遍历所有边，收集出现在有效边中的所有节点（排除 relation 为 '无' 的边）
        for edge in story:
            if edge['relation'] != '无':  # 排除 relation 为 '无' 的边
                all_connected_nodes.add(edge['first_event'])
                all_connected_nodes.add(edge['second_event'])

        # 孤立节点是出现在事件列表中，但没有出现在任何有效边中的节点
        isolated_nodes = [event for event in events if event not in all_connected_nodes]



        
        for event in isolated_nodes:
            node = pydot.Node(f"{events_dict[event]}\n{event}")
            isolated_graph.add_node(node)

        # 保存主图文件
        main_filename = f"/disk/NarGINA/serve/asset/{output_file_name}_main_graph_{count}.png"
        main_graph.write_png(main_filename, encoding='utf-8')
        print(f"Saved main graph for story {count} as {main_filename}")

        # 保存孤立节点图文件
        isolated_filename = f"/disk/NarGINA/serve/asset/{output_file_name}_isolated_graph_{count}.png"
        isolated_graph.write_png(isolated_filename, encoding='utf-8')
        print(f"Saved isolated graph for story {count} as {isolated_filename}")

        return main_filename,isolated_filename
    

def image_to_plotly(img_path, max_width=1500, max_height=500,is_gold=False):
    # 打开图片
    image = Image.open(img_path)
    img_width, img_height = image.size

    # 计算缩放比例
    width_ratio = max_width / img_width
    height_ratio = max_height / img_height
    scale = min(width_ratio, height_ratio)  # 使用较小的比例以确保图片完全显示在限制内

    # 计算缩放后的宽高
    scaled_width = img_width * scale
    scaled_height = img_height * scale

    # 创建Figure对象
    fig = go.Figure()



    # 设置坐标轴和布局，以确保图片完全显示
    t = max(scaled_width,scaled_height)
    fig.update_xaxes(visible=False, range=[0, t])
    fig.update_yaxes(visible=False, range=[0, t])
    fig.update_layout(
        width=t,
        height=t,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',  # 图表区域的背景颜色
        paper_bgcolor='white'  # 整个纸张的背景颜色
    )
    # 添加图片作为背景图层
    fig.add_layout_image(
        dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=t-500,  # 确保图片位于正确的位置
            sizex=scaled_width,
            sizey=scaled_height,
            sizing="stretch",  # 拉伸图片以适应新的尺寸
            opacity=1,
            layer="below"
        )
    )
    return fig    # 导出为HTML文件
    html_path = "/disk/NarGINA/serve/asset/image_display_fullsize.html"
    fig.write_html(html_path)
    with open(html_path, "r") as file:
        return file.read()  # 你可以改为读取具体的HTML文件或内容
    


def gold_image_to_plotly(img_path, max_width=1500, max_height=1000):
    # 打开图片
    image = Image.open(img_path)
    img_width, img_height = image.size

    # 计算缩放比例
    width_ratio = max_width / img_width
    height_ratio = max_height / img_height
    scale = min(width_ratio, height_ratio)  # 使用较小的比例以确保图片完全显示在限制内

    # 计算缩放后的宽高
    scaled_width = img_width * scale
    scaled_height = img_height * scale
    scaled_height = scaled_height*2
    # 创建Figure对象
    fig = go.Figure()



    # 设置坐标轴和布局，以确保图片完全显示
    t = max(scaled_width,scaled_height)
    fig.update_xaxes(visible=False, range=[0, t])
    fig.update_yaxes(visible=False, range=[0, scaled_height*1.5])
    fig.update_layout(
        width=t,
        height=scaled_height*1.5,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',  # 图表区域的背景颜色
        paper_bgcolor='white'  # 整个纸张的背景颜色
    )
    # 添加图片作为背景图层
    fig.add_layout_image(
        dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=scaled_height*1.5,  # 确保图片位于正确的位置
            sizex=scaled_width,
            sizey=scaled_height,
            sizing="stretch",  # 拉伸图片以适应新的尺寸
            opacity=1,
            layer="below"
        )
    )
    return fig    # 导出为HTML文件
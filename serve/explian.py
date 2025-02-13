import random
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pydot

# 加载编码器模型
encoder = SentenceTransformer('/disk/NarGINA/weights/sentence-transformers')
relation_descriptions = [
    "使能-因果",
    "动机-因果",
    "物理-因果",
    "心理-因果",
    "并列",
    "无",
    "事件"
]

# 定义图数据编码函数
def graph_json_2_pyg(g_json, encoder):
    edge_embeddings = torch.from_numpy(encoder.encode(relation_descriptions)).cpu()

    node_dict = {}  # 存储节点和索引
    edges = []  # 存储边
    edge_types = []  # 存储边类型索引
    edge_attr_list = []  # 存储边特征向量
    node_texts = []  # 存储节点文本

    # 分配索引并创建边，收集节点文本
    for graph in g_json:
        for edge in graph:
            first_event = edge["first_event"]
            second_event = edge["second_event"]
            relation = edge["relation"]

            # 分配节点索引
            if first_event not in node_dict:
                node_dict[first_event] = len(node_dict)
                node_texts.append(first_event)
            if second_event not in node_dict:
                node_dict[second_event] = len(node_dict)
                node_texts.append(second_event)

            # 添加边和边类型
            if relation != "无":
                first_idx = node_dict[first_event]
                second_idx = node_dict[second_event]
                edges.append([first_idx, second_idx])

                edge_type = relation_descriptions.index(relation)
                edge_types.append(edge_type)
                edge_attr_list.append(edge_embeddings[edge_type])

    # 创建边索引张量
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.stack(edge_attr_list) if edge_attr_list else None

    edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
    node_embeddings = torch.from_numpy(encoder.encode(node_texts)).cpu()
    data = Data(x=node_embeddings, edge_index=edge_index, edge_type=edge_type_tensor, edge_attr=edge_attr)

    return data, node_texts, node_dict


# 加载 JSON 文件数据
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 筛选关键事件节点
def filter_key_events(gold_data, gold_node_dict, threshold=2):
    key_events = set()
    in_degrees = {node: 0 for node in gold_node_dict.keys()}
    out_degrees = {node: 0 for node in gold_node_dict.keys()}

    for src, dst in gold_data.edge_index.t().tolist():
        src_node = list(gold_node_dict.keys())[src]
        dst_node = list(gold_node_dict.keys())[dst]
        out_degrees[src_node] += 1
        in_degrees[dst_node] += 1

    for node, degree in in_degrees.items():
        if degree >= threshold or out_degrees[node] >= threshold:
            key_events.add(node)

    return key_events


# 更新叙事图的节点文本，添加缺失节点
def update_narrative_with_missing_nodes(narrative_texts, narrative_node_dict, missing_nodes):
    updated_texts = narrative_texts[:]
    updated_node_dict = {text: idx for idx, text in enumerate(updated_texts)}

    for missing_node in missing_nodes:
        if missing_node not in updated_node_dict:
            updated_texts.append(f"[缺失] {missing_node}")
            updated_node_dict[f"[缺失] {missing_node}"] = len(updated_texts) - 1

    return updated_texts, updated_node_dict


# 查找缺失边
def find_missing_edges(updated_texts, gold_data, node_mapping, updated_node_dict):
    missing_edges = []
    recorded_edges = set()

    for i in range(gold_data.edge_index.size(1)):
        gold_src = list(gold_data.edge_index[0])[i].item()
        gold_dst = list(gold_data.edge_index[1])[i].item()
        gold_relation = gold_data.edge_type[i].item()

        gold_src_node = list(updated_node_dict.keys())[gold_src]
        gold_dst_node = list(updated_node_dict.keys())[gold_dst]

        if gold_src_node in updated_node_dict and gold_dst_node in updated_node_dict:
            src_idx = updated_node_dict[gold_src_node]
            dst_idx = updated_node_dict[gold_dst_node]
            if (src_idx, dst_idx) not in recorded_edges:
                missing_edges.append({
                    "source": gold_src_node,
                    "target": gold_dst_node,
                    "relation": gold_relation
                })
                recorded_edges.add((src_idx, dst_idx))

    return missing_edges


def find_wrong_edges(narrative_data, gold_data, node_mapping, narrative_node_dict, gold_node_dict):
    """
    查找叙事图中关系错误的边。
    """
    wrong_edges = []  # 关系错误的边

    # 转换金标图的边集合为方便查询的格式
    gold_edge_set = {
        (gold_data.edge_index[0, i].item(), gold_data.edge_index[1, i].item()): gold_data.edge_type[i].item()
        for i in range(gold_data.edge_index.size(1))
    }

    # 遍历叙事图的每一条边
    for i in range(narrative_data.edge_index.size(1)):
        narrative_src = narrative_data.edge_index[0, i].item()
        narrative_dst = narrative_data.edge_index[1, i].item()
        narrative_relation = narrative_data.edge_type[i].item()

        # 获取叙事图中的源节点和目标节点文本
        narrative_src_node = list(narrative_node_dict.keys())[narrative_src]
        narrative_dst_node = list(narrative_node_dict.keys())[narrative_dst]

        # 映射到金标图中的节点
        mapped_src_nodes = [
            gold_node_dict.get(node) for node, mapped_nodes in node_mapping.items() if node == narrative_src_node
        ]
        mapped_dst_nodes = [
            gold_node_dict.get(node) for node, mapped_nodes in node_mapping.items() if node == narrative_dst_node
        ]

        # 检查映射后的边是否在金标图中
        is_wrong_edge = True
        for gold_src in mapped_src_nodes:
            for gold_dst in mapped_dst_nodes:
                if (gold_src, gold_dst) in gold_edge_set:
                    gold_relation = gold_edge_set[(gold_src, gold_dst)]
                    if gold_relation == narrative_relation:
                        is_wrong_edge = False  # 关系匹配，非错误边
                        break
                if not is_wrong_edge:
                    break

        # 如果未找到匹配的关系类型，记录为错误边
        if is_wrong_edge:
            wrong_edges.append({
                "source": narrative_src_node,
                "target": narrative_dst_node,
                "gold_relation": gold_relation if 'gold_relation' in locals() else -2,
                "narrative_relation": narrative_relation
            })

    return wrong_edges

def connect_missing_nodes(narrative_texts, updated_texts, missing_nodes, narrative_data, threshold=0.7):
    """
    自动为缺失节点与叙事图中最近的节点建立连接。
    """
    missing_edges = []

    # 计算叙事图节点和缺失节点的嵌入
    narrative_embeddings = torch.from_numpy(encoder.encode(narrative_texts)).cpu()
    missing_embeddings = torch.from_numpy(encoder.encode([node for node in updated_texts if "[缺失]" in node])).cpu()

    # 为每个缺失节点寻找最相似的叙事图节点
    for missing_idx, missing_embedding in enumerate(missing_embeddings):
        best_match = None
        best_similarity = threshold  # 只有高于阈值才添加边

        for narrative_idx, narrative_embedding in enumerate(narrative_embeddings):
            similarity = cosine_similarity(
                missing_embedding.unsqueeze(0).numpy(),
                narrative_embedding.unsqueeze(0).numpy()
            )[0][0]

            if similarity > best_similarity:
                best_match = narrative_idx
                best_similarity = similarity

        # 如果找到匹配，添加辅助边
        if best_match is not None:
            missing_edges.append({
                "source": updated_texts[len(narrative_texts) + missing_idx],  # 缺失节点文本
                "target": narrative_texts[best_match],  # 最匹配的叙事图节点文本
                "relation": -1  # 特殊关系类型表示辅助边
            })

    return missing_edges


# 可视化叙事图、缺失边和关系错误边
def visualize_with_missing_and_wrong_edges(narrative_data, updated_texts, missing_edges, wrong_edges, relation_descriptions, save_path=None):
    graph = pydot.Dot(graph_type='digraph', rankdir='LR')
    # 设置画布大小

    # 记录已经绘制的边，以防重复绘制
    drawn_edges = set()
    connected_nodes = set()  # 用于记录与任何边相连的节点
    # 添加关系错误的边（橙色加粗实线，优先绘制）
    for edge in wrong_edges:
        if edge["source"] in updated_texts and edge["target"] in updated_texts:
            src_idx = updated_texts.index(edge["source"])
            dst_idx = updated_texts.index(edge["target"])
            gold_relation = relation_descriptions[edge["gold_relation"]] if edge["gold_relation"] is not None else "Unknown"
            narrative_relation = relation_descriptions[edge["narrative_relation"]]
            edge_label = f"冗余: {narrative_relation}"
            edge = pydot.Edge(
                str(src_idx), str(dst_idx),
                label=edge_label,
                color="blue",  # 橙色
                fontcolor="blue",  # 标签字体颜色
                penwidth=3
            )
            graph.add_edge(edge)
            # 将边加入已绘制集合
            drawn_edges.add((src_idx, dst_idx))
            # 记录连接过的节点
            connected_nodes.add(src_idx)
            connected_nodes.add(dst_idx)

    # 添加缺失边（红色加粗虚线）
    for edge in missing_edges:
        if edge["source"] in updated_texts and edge["target"] in updated_texts:
            src_idx = updated_texts.index(edge["source"])
            dst_idx = updated_texts.index(edge["target"])
            #
            relation_type = relation_descriptions[edge["relation"]]
            edge_label = f"缺失: {relation_type}"
            edge = pydot.Edge(
                str(src_idx), str(dst_idx),
                label=edge_label,
                color="#FF0000",   # 红色
                fontcolor="#FF0000",  # 标签字体颜色一致
                penwidth=3,
                style="dashed"
            )
            graph.add_edge(edge)
            # 将边加入已绘制集合
            #drawn_edges.add((src_idx, dst_idx))
            # 记录连接过的节点
            connected_nodes.add(src_idx)
            connected_nodes.add(dst_idx)

    # 添加叙事图中的边（如果对应的两个节点之间没有绘制过错误边）
    for i, (src, dst) in enumerate(narrative_data.edge_index.t().tolist()):
        if (src, dst) not in drawn_edges:
            relation_type = narrative_data.edge_type[i].item()
            relation_label = relation_descriptions[relation_type]
            edge = pydot.Edge(
                str(src), str(dst),
                label=relation_label,
                color="black",  # 深灰色
                fontcolor="black",  # 标签字体颜色
                penwidth=1.0
            )
            graph.add_edge(edge)
            # 记录连接过的节点
            connected_nodes.add(src)
            connected_nodes.add(dst)

    # 添加节点，但仅添加那些与边相连的节点（包括缺失节点）
    for idx, text in enumerate(updated_texts):
        if idx in connected_nodes or text.startswith("[缺失]"):  # 仅添加与边相连的节点或缺失节点
            if text.startswith("[缺失]"):
                node = pydot.Node(
                    name=str(idx), label=text, style="filled", fillcolor="yellow", shape="box"
                )
            else:
                node = pydot.Node(
                    name=str(idx), label=str(idx) + "\n" + text, style="filled", fillcolor="lightblue", shape="ellipse"
                )
            graph.add_node(node)

    # 保存或展示图
    if save_path:
        graph.write_png(save_path)
        print(f"Graph saved to {save_path}")
    else:
        graph.write_png("output_graph.png")
        print("Graph saved to output_graph.png")

def filter_disconnected_nodes(narrative_data, updated_texts, missing_edges, wrong_edges):
    """
    过滤没有任何边相连的孤立节点，但保留缺失节点。
    """
    connected_nodes = set()

    # 遍历叙事图中的边
    for src, dst in narrative_data.edge_index.t().tolist():
        connected_nodes.add(src)
        connected_nodes.add(dst)

    # 遍历缺失边
    for edge in missing_edges:
        if edge["source"] in updated_texts and edge["target"] in updated_texts:
            connected_nodes.add(updated_texts.index(edge["source"]))
            connected_nodes.add(updated_texts.index(edge["target"]))

    # 遍历错误边
    for edge in wrong_edges:
        if edge["source"] in updated_texts and edge["target"] in updated_texts:
            connected_nodes.add(updated_texts.index(edge["source"]))
            connected_nodes.add(updated_texts.index(edge["target"]))

    # 添加缺失节点（以 "[缺失]" 开头的节点必须保留）
    filtered_texts = [
        text for idx, text in enumerate(updated_texts)
        if idx in connected_nodes or text.startswith("[缺失]")
    ]

    return filtered_texts


def match_nodes_and_edges(narrative_data, gold_data, narrative_texts, gold_texts, narrative_node_dict, gold_node_dict):
    node_mapping = {}  # 存储每个叙事图节点对应的多个金标图节点
    threshold = 0.63  # 相似度阈值
    narrative_embeddings = narrative_data.x
    gold_embeddings = gold_data.x

    # 节点匹配（支持一对多）
    for i, narrative_embedding in enumerate(narrative_embeddings):
        matching_nodes = []
        for j, gold_embedding in enumerate(gold_embeddings):
            similarity = cosine_similarity(
                narrative_embedding.unsqueeze(0).numpy().reshape(1, -1),
                gold_embedding.unsqueeze(0).numpy().reshape(1, -1)
            )[0][0]
            if similarity >= threshold:
                gold_node = list(gold_node_dict.keys())[j]
                matching_nodes.append(gold_node)

        # 存储该叙事图节点的所有匹配金标节点
        if matching_nodes:
            narrative_node = list(narrative_node_dict.keys())[i]
            node_mapping[narrative_node] = matching_nodes

    # 转换金标图的边为集合形式以便快速匹配
    gold_edge_set = {
        (gold_data.edge_index[0, i].item(), gold_data.edge_index[1, i].item(), gold_data.edge_type[i].item())
        for i in range(gold_data.edge_index.size(1))
    }

    # 边匹配
    common_edges = []
    for idx, edge in enumerate(narrative_data.edge_index.t().tolist()):
        src, dst = edge
        src_node = list(narrative_node_dict.keys())[src]
        dst_node = list(narrative_node_dict.keys())[dst]

        if src_node in node_mapping and dst_node in node_mapping:
            # 对每对可能的匹配节点进行边的验证
            for mapped_src in [gold_node_dict.get(node) for node in node_mapping[src_node]]:
                for mapped_dst in [gold_node_dict.get(node) for node in node_mapping[dst_node]]:
                    edge_type = narrative_data.edge_type[idx].item()
                    if (mapped_src, mapped_dst, edge_type) in gold_edge_set:
                        common_edges.append((src_node, dst_node, edge_type))

    return node_mapping, common_edges


def find_missing_nodes(gold_node_dict, node_mapping):
    """
    找出金标图中未被匹配的节点。
    """
    # 获取所有金标图的节点
    gold_nodes = set(gold_node_dict.keys())

    # 获取所有被映射到的金标节点
    mapped_gold_nodes = set()
    for _, mapped_nodes in node_mapping.items():
        mapped_gold_nodes.update(mapped_nodes)

    # 找出未被映射的金标节点
    missing_nodes = gold_nodes - mapped_gold_nodes
    return list(missing_nodes)
# 主函数
def explain_visualization(narrative_json, gold_json, diff_save_path):
    # narrative_json = load_json(narrative_json_path)
    # gold_json = load_json(gold_json_path)

    narrative_data, narrative_texts, narrative_node_dict = graph_json_2_pyg(narrative_json, encoder)
    gold_data, gold_texts, gold_node_dict = graph_json_2_pyg(gold_json, encoder)

    key_events = filter_key_events(gold_data, gold_node_dict, threshold=2)
    filtered_gold_node_dict = {node: idx for idx, node in enumerate(key_events)}
    filtered_gold_edges = [
        (src, dst, etype)
        for src, dst, etype in zip(
            gold_data.edge_index[0].tolist(),
            gold_data.edge_index[1].tolist(),
            gold_data.edge_type.tolist()
        )
        if list(gold_node_dict.keys())[src] in key_events and list(gold_node_dict.keys())[dst] in key_events
    ]

    filtered_gold_data = Data(
        x=gold_data.x[list(filtered_gold_node_dict.values())],
        edge_index=torch.tensor([[filtered_gold_node_dict[list(gold_node_dict.keys())[src]],
                                  filtered_gold_node_dict[list(gold_node_dict.keys())[dst]]]
                                 for src, dst, _ in filtered_gold_edges]).t().contiguous(),
        edge_type=torch.tensor([etype for _, _, etype in filtered_gold_edges])
    )

    node_mapping, common_edges = match_nodes_and_edges(
        narrative_data, filtered_gold_data, narrative_texts, gold_texts, narrative_node_dict, filtered_gold_node_dict
    )

    missing_nodes = find_missing_nodes(filtered_gold_node_dict, node_mapping)
    updated_texts, updated_node_dict = update_narrative_with_missing_nodes(narrative_texts, narrative_node_dict, missing_nodes)

    missing_edges = find_missing_edges(updated_texts, filtered_gold_data, node_mapping, updated_node_dict)
    wrong_edges = find_wrong_edges(narrative_data, filtered_gold_data, node_mapping, narrative_node_dict, filtered_gold_node_dict)

    additional_missing_edges = connect_missing_nodes(
        narrative_texts, updated_texts, missing_nodes, narrative_data, threshold=0.5
    )

    #filtered_texts = filter_disconnected_nodes(narrative_data, updated_texts, missing_edges+additional_missing_edges, wrong_edges)


    # 随机删除 40% 的缺失边和错误边
    missing_edges = random.sample(missing_edges, int(len(missing_edges) * 0.6)) if len(missing_edges) > 0 else []
    wrong_edges = random.sample(wrong_edges, int(len(wrong_edges) * 0.5)) if len(wrong_edges) > 0 else []


    visualize_with_missing_and_wrong_edges(
        narrative_data, updated_texts, missing_edges + additional_missing_edges, wrong_edges, relation_descriptions, save_path=diff_save_path
    )

if __name__ == "__main__":
    # 示例文件路径（请替换为实际路径）
    narrative_json_path = '/disk/NarGINA/ChildText/gold_data/content.json'
    gold_json_path = '/disk/NarGINA/ChildText/gold_data/gold_graph_content.json'
    narrative_save_path = '/disk/NarGINA/serve/asset/narrative_graph.png'
    diff_save_path = '/disk/NarGINA/serve/asset/difference_graph.png'

    explain_visualization(narrative_json_path, gold_json_path, diff_save_path)





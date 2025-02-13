# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import graphviz
from matplotlib import pyplot as plt
import pydot
import pydot
import networkx as nx
from networkx.drawing.nx_pydot import from_pydot
from pyvis.network import Network
# plt.rcParams['font.family'] = 'SimHei' # 设置字体为黑体

# plt.rcParams['axes.unicode_minus'] = False # 正确显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号
from pyvis.network import Network

# def get_graph(content):
#     label_dict = {"动机-因果": 0, "心理-因果": 1, "物理-因果": 2, "使能-因果": 3, "并列": 4, "无": 5}
#     relation_colors = {
#         '动机-因果': 'red',
#         '心理-因果': 'blue',
#         '物理-因果': 'green',
#         "使能-因果": 'orange',
#         "并列": 'purple',
#         '无': 'black'
#     }
    
#     for count, story in enumerate(content):
#         # Initialize the graph
#         net = Network(directed=True, height="800px", width="100%", bgcolor="#ffffff", font_color="black",)
#         #net.force_atlas_2based()  # Enable force-directed layout

#         # Extract unique events
#         events = []
#         for edge in story:
#             if edge['first_event'] not in events:
#                 events.append(edge['first_event'])
#             if edge['second_event'] not in events:
#                 events.append(edge['second_event'])

#         # Track nodes connected by valid relations
#         connected_nodes = set()
#         for edge in story:
#             if edge['relation'] != '无':  # Only consider valid edges
#                 connected_nodes.add(edge['first_event'])
#                 connected_nodes.add(edge['second_event'])

#         # Determine isolated nodes
#         isolated_nodes = [event for event in events if event not in connected_nodes]

#         # Add connected nodes and edges
#         for edge in story:
#             first_event = edge['first_event']
#             second_event = edge['second_event']
#             relation = edge['relation']
#             color = relation_colors[relation]

#             if relation != '无':
#                 net.add_node(first_event, label=first_event)
#                 net.add_node(second_event, label=second_event)
#                 net.add_edge(first_event, second_event, title=relation, color=color)

#         # # Add isolated nodes with independent positions
#         # for i, event in enumerate(isolated_nodes):
#         #     net.add_node(event, label=event, x=(i + 1) * 100, y=-1000)  # Spread nodes horizontally

#         # Save the graph as an HTML file
#         filename = f"story_graph_{count}.html"
#         net.save_graph(filename)
#         print(f"Saved graph for story {count} as {filename}")



def get_graph(content):
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
        main_graph = pydot.Dot(graph_type='digraph', encoding='utf-8',prog='dot')
        main_graph.set_edge_defaults(fontname='SimHei')
        main_graph.set_node_defaults(fontname='SimHei')
        
        # 设置垂直布局
        main_graph.set('rankdir', 'TB')

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
        main_filename = f"/disk/NarGINA/serve/asset/_main_graph_{count}.png"
        main_graph.write_png(main_filename, encoding='utf-8')
        print(f"Saved main graph for story {count} as {main_filename}")
        main_graph.write_dot("graph.dot")
        # 保存孤立节点图文件
        isolated_filename = f"/disk/NarGINA/serve/asset/_isolated_graph_{count}.png"
        isolated_graph.write_png(isolated_filename, encoding='utf-8')
        print(f"Saved isolated graph for story {count} as {isolated_filename}")
        dot_data = main_graph.create_dot()
        dot_string = dot_data.decode('utf-8')
        layouted_graph = pydot.graph_from_dot_data(dot_string)
        #将 pydot 图转换为 networkx 图
        nx_graph = from_pydot(layouted_graph[0])
        nx_graph = nx.DiGraph(nx_graph)
        # 提取节点的位置信息
        pos = {}
        for node, data in nx_graph.nodes(data=True):
            if 'pos' in data:
                try:
                    # 提取 pos 并解析为 (x, y) 坐标
                    x, y = map(float, data['pos'].strip('!').split(','))
                    pos[node] = (x, y)
                except ValueError:
                    print(f"节点 {node} 的位置解析失败: {data['pos']}")

        # 检查 pos 是否为空
        if not pos:
            print("没有找到节点位置，检查 pydot 图是否包含 pos 属性。")
        else:
            print("节点位置:", pos)
        # 可视化 networkx 图
        import matplotlib.pyplot as plt
        nx.draw(nx_graph, pos=pos, with_labels=True)
        plt.show()
        plt.savefig("pydot2networkx.png")
            

# pred[]中存放pred-class数据
with open('/disk/NarGINA/relation_predictions.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    pred = []
    tmp = []
    for line in lines:
        if 'story' in line:
            pred.append(tmp)
            tmp = []
            # continue
        else:
            tmp.append(line.split('predicted class: ')[1].split(', ')[0])
            # pred.append(label_dict[line.split('predicted class: ')[1].split(', ')[0]])
    pred.append(tmp)
    # 移除pred列表中的第一个元素（空tmp）
    pred.pop(0)


# content中存储事件1和事件2的关系（relation）
with open('/disk/NarGINA/dev_relations.txt', 'r',encoding='utf-8') as file:
    # with open('dev_relations.txt','r',encoding='utf-8') as file:
    lines = file.readlines()
    content = []
    tmp = []
    count = 0
    for story in pred:

        for index in range(len(story)):
            e = lines[index + count].split(' ')[0]
            edge = {
                'first_event': e.split('<second_event>')[0].split('<first_event>')[1],
                'second_event': e.split('<second_event>')[1],
                'relation': story[index]
            }
            tmp.append(edge)

        count = index
        content.append(tmp)

print('generate pictures.....')
get_graph(content)
print('finishing generate.....')
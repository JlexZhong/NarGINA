import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import graphviz
import pydot
import plotly.graph_objects as go
import networkx as nx

def get_graph(content):
    label_dict = {"动机-因果": 0, "心理-因果": 1, "物理-因果": 2, "使能-因果": 3, "并列": 4, "无": 5}
    # Define edge colors dictionary
    relation_colors = {
        '动机-因果': 'red',
        '心理-因果': 'blue',
        '物理-因果': 'green',
        "使能-因果": 'orange',
        "并列": 'purple',
        '无': 'black'
    }
    count = -1

    for story in content:
        count += 1
        events = []
        for edge in story:
            if edge['first_event'] not in events:
                events.append(edge['first_event'])
            if edge['second_event'] not in events:
                events.append(edge['second_event'])

        events_dict = {event: idx for idx, event in enumerate(events)}

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes and edges to the graph
        for edge in story:
            first_event_id = events_dict[edge['first_event']]
            second_event_id = events_dict[edge['second_event']]
            relation = edge['relation']

            color = relation_colors[relation]

            if relation != '无':
                G.add_edge(first_event_id, second_event_id, label=relation, color=color)

        # Set positions for nodes using spring layout
        pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_colors = []
        edge_texts = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_colors.append(edge[2]['color'])
            edge_texts.append(edge[2]['label'])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color='black'),
            hoverinfo='text',
            mode='lines',
            text=edge_texts
        )

        # Create node traces
        node_x = []
        node_y = []
        node_texts = []

        for node, (x, y) in pos.items():
            node_x.append(x)
            node_y.append(y)
            node_texts.append(str(node) + "\n" + events[node])

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_texts,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=10,
                color='lightblue',
                line_width=2
            )
        )

        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Graph Visualization',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

        # Show the figure
        fig.show()


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
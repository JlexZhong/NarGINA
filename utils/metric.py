import re
import jieba
import spacy



# 计算依存树的深度（最大深度和平均深度）
def calculate_dependency_tree_depths(doc):
    max_depth = 0
    total_avg_depth = 0  # 累积所有句子的平均深度
    sentence_count = 0

    def traverse(node, depth):
        nonlocal current_max_depth
        nonlocal current_total_depth
        if depth > current_max_depth:
            current_max_depth = depth
        current_total_depth += depth
        for child in node.children:
            traverse(child, depth + 1)

    for sent in doc.sents:
        sentence_count += 1
        root = [token for token in sent if token.head == token][0]  # 找到句子的根节点
        current_max_depth = 0  # 每个句子的最大深度
        current_total_depth = 0  # 每个句子的总深度
        token_count = len([token for token in sent])  # 句子的 token 数量
        
        # 计算句子的依存树深度
        traverse(root, 1)  # 从1开始计数
        total_avg_depth += (current_total_depth / token_count)  # 将该句子的平均深度累加

        if current_max_depth > max_depth:  # 更新最大深度
            max_depth = current_max_depth

    # 计算总的平均深度
    average_depth = total_avg_depth / sentence_count if sentence_count > 0 else 0

    return max_depth, average_depth

# 统计从属连词的数量
def count_connectives(text):
        # 常见的从属连词或关联词
    connectives = ["因为", "所以", "虽然", "但是", "因此", "即使", "然而", "而且", "既然", "除非", "尽管", "还是","与此同时","同时"]

    count = 0
    for conn in connectives:
        count += text.count(conn)
    return count

# 定义统计函数
def analyze_composition(text,nlp):

    #     # 加载中文预训练模型（zh_core_web_sm）
    # nlp = spacy.load("zh_core_web_sm")


    # 分句（通过标点符号分句，考虑到中英文符号）
    sentences = re.split(r'[。！？!?\.]', text)
    sentences = [sent.strip() for sent in sentences if sent.strip()]  # 去除空白句子

    # 分词
    words = jieba.lcut(text)
    total_word_count = len(words)  # 总词汇数
    unique_words = set(words)  # 不同词汇
    unique_word_count = len(unique_words)

    # 不同词汇出现率 (TTR)
    ttr = unique_word_count / total_word_count if total_word_count > 0 else 0

    # 句子总数
    total_sentences = len(sentences)

    # 平均话语长度 (MLU)
    mlu = total_word_count / total_sentences if total_sentences > 0 else 0

    # 使用 Spacy 计算依存树的最大深度和平均深度
    doc = nlp(text)
    max_syntax_depth, avg_syntax_depth = calculate_dependency_tree_depths(doc)

    # 统计从属连词或关联词的使用次数
    connective_count = count_connectives(text)

    # 生成描述性文本
    result_text = f"这篇作文共有 {total_word_count} 个词汇，不同词汇出现率为 {ttr:.2f}；共有 {total_sentences} 个句子，平均每个句子有 {mlu:.1f} 个词汇；体现句法复杂度的最大依存树深度为 {max_syntax_depth}，平均依存树深度为 {avg_syntax_depth:.1f}；文中还包含了 {connective_count} 次从属连词或关联词的使用。"

    return result_text


# # 测试
# text = "从前有一个男孩，他有一只狗和一只宠物青蛙。 他把青蛙放在他卧室的一个大罐子里。一天晚上，当他和他的狗在睡觉时，青蛙从罐子里爬了出来。 它从一扇打开的窗口跳了出去。第二天早上，当男孩和狗醒来时，他们发现罐子是空的。小男孩到处找青蛙。小狗也找青蛙。当狗试图往罐子里看时，它的头被卡住了。男孩从开着的窗户喊道：“青蛙，你在哪里？”。狗探出窗外，罐子仍然卡在它的头上。罐子太重了，小狗头朝下从窗户掉了下去。罐子摔碎了。小男孩跳下窗台很生气的抱着小狗 男孩和小狗在外面找青蛙。男孩呼喊着青蛙。当小狗对着蜂巢里的蜜蜂叫的时候，小男孩朝地洞里喊。一只地鼠从洞里钻出来，正好咬在男孩的鼻子上。与此同时，小狗还在骚扰蜜蜂，跳到树上对它们叫。蜂窝掉下来，所有的蜜蜂都飞出来了。小男孩一点儿都没有注意到小狗。他注意到树上有一个大洞。于是他爬上树，朝洞里喊。突然，一只猫头鹰从洞里猛扑出来，把男孩撞倒在地。小狗以最快的速度从男孩身边跑过，因为蜜蜂在追它。猫头鹰一路追着小男孩，追到一块大石头前。小男孩爬上那块石头，再次呼喊着青蛙。他抓住一些树枝，这样他就不会掉下去。但是树枝并不是真的树枝，而是鹿角。鹿把小男孩放在头上。鹿开始奔跑，小男孩还在它头上。小狗也跟着跑。它们越来越接近悬崖。鹿突然停了下来，男孩和狗从悬崖边掉了下去。悬崖下面有一个池塘。他们一个接一个地扑通落进了池塘。他们听到了熟悉的声音。男孩让小狗安静一点。他们悄悄爬上去，朝一根大原木后面看去。他们翻过原木，看见两只大青蛙。还有一些小青蛙，其中一只跳向男孩。小男孩把一只小青蛙带回家，并开心地和其他青蛙说再见。"

# result = analyze_composition(text)
# print(result)

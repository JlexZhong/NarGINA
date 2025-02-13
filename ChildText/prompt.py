DEFAULT_GRAPH_TOKEN = "<graph>"



GOLD_TEXT = """从前有一个男孩，他有一只狗和一只宠物青蛙。 他把青蛙放在他卧室的一个大罐子里。一天晚上，当他和他的狗在睡觉时，青蛙从罐子里爬了出来。 它从一扇打开的窗口跳了出去。第二天早上，当男孩和狗醒来时，他们发现罐子是空的。小男孩到处找青蛙。小狗也找青蛙。当狗试图往罐子里看时，它的头被卡住了。男孩从开着的窗户喊道：“青蛙，你在哪里？”。狗探出窗外，罐子仍然卡在它的头上。罐子太重了，小狗头朝下从窗户掉了下去。罐子摔碎了。小男孩跳下窗台很生气的抱着小狗 男孩和小狗在外面找青蛙。男孩呼喊着青蛙。当小狗对着蜂巢里的蜜蜂叫的时候，小男孩朝地洞里喊。一只地鼠从洞里钻出来，正好咬在男孩的鼻子上。与此同时，小狗还在骚扰蜜蜂，跳到树上对它们叫。蜂窝掉下来，所有的蜜蜂都飞出来了。小男孩一点儿都没有注意到小狗。他注意到树上有一个大洞。于是他爬上树，朝洞里喊。突然，一只猫头鹰从洞里猛扑出来，把男孩撞倒在地。小狗以最快的速度从男孩身边跑过，因为蜜蜂在追它。猫头鹰一路追着小男孩，追到一块大石头前。小男孩爬上那块石头，再次呼喊着青蛙。他抓住一些树枝，这样他就不会掉下去。但是树枝并不是真的树枝，而是鹿角。鹿把小男孩放在头上。鹿开始奔跑，小男孩还在它头上。小狗也跟着跑。它们越来越接近悬崖。鹿突然停了下来，男孩和狗从悬崖边掉了下去。悬崖下面有一个池塘。他们一个接一个地扑通落进了池塘。他们听到了熟悉的声音。男孩让小狗安静一点。他们悄悄爬上去，朝一根大原木后面看去。他们翻过原木，看见两只大青蛙。还有一些小青蛙，其中一只跳向男孩。小男孩把一只小青蛙带回家，并开心地和其他青蛙说再见。"""


SCORE_CRITERIA="""- 1分：叙事几乎无法理解，没有清晰的开端、发展或结尾，因果关系完全缺失，词汇极其有限，句子基本无法构成有效的内容，语法和拼写错误严重且频繁。
- 2分：叙事模糊，缺乏开端、发展或结尾，因果关系缺失，词汇极为有限，句子非常简单且重复，语法和拼写错误频繁。
- 3分：叙事开始出现但不完整，因果关系偶有出现，词汇仍然基础，句子简单且重复，语法和拼写错误较多。
- 4分：叙事基本清晰但缺乏细节，因果关系较弱，词汇稍有提升，句子结构简单，语法和拼写错误减少。
- 5分：叙事完整，开端、发展和结尾清晰，因果关系合理但简单，词汇适中，句子多为简单句，语法和拼写大体正确。
- 6分：叙事连贯但缺乏细节，因果关系明确，词汇丰富度有所提升，句子结构多样，语法和拼写总体正确。
- 7分：叙事详细且连贯，因果关系自然，词汇较丰富且生动，句子结构多样，语法和拼写几乎无误。
- 8分：叙事非常详细且生动，因果关系清晰，词汇丰富且精准，句子结构复杂，语法和拼写几乎没有错误。
- 9分：叙事极为详细，逻辑紧密且富有层次，词汇非常丰富，句子结构高度复杂，语法和拼写完美无误。
- 10分：叙事极其详细、富有创意，因果关系完美，词汇极其丰富且富有文学性，句子结构复杂优美，语法和拼写无瑕疵。"""


GRAPH_RELATIONS_DESC = """
- 动机-因果关系：事件1提供了促使事件2发生的目标或动机。事件1通常包含与目标相关的信息。
- 心理-因果关系：事件1引发了事件2中的内部反应，如欲望、信念、想法、意图或情绪，形成一种心理上的因果联系。
- 物理-因果关系：事件1通过物理或机械过程直接导致事件2的发生，不需要额外的上下文或背景故事。
- 使能-因果关系：事件1是事件2发生的必要条件，但单凭事件1不足以导致事件2发生。事件1充当了一个条件，而非直接原因。
- 并列关系：事件1和事件2同时发生或在时间上重叠，表示一种同步或共时关系。"""



###############################GRAPH MATCH #######################

PROMPT_TEMPLATE_GRAPH_MATCH = \
f"""给一个从儿童叙事文中提取的叙事图，表示一系列事件及其相互关系。图中的节点代表事件，格式为：谓语(主语;宾语;时间状语;地点状语)。边表示事件之间的关系，包括并列、动机-因果、物理-因果、心理-因果和使能-因果。请利用叙事图信息来理解事件之间的关系和叙事顺序。
通过对叙事图进行随机采样得到子图，在给定叙事子图中，有节点token序列：{DEFAULT_GRAPH_TOKEN}
打乱顺序的事件文本：
<shuffled_text_sequence>
Question:请根据节点token序列的顺序对事件文本进行重新排序（即完成节点token和事件文本的匹配）"""

##############################TRAIN PROMPT ###########################################

PROMPT_TEMPLATE = f"""你的任务是儿童叙事能力评价，主题是“青蛙，你在哪？”。

### 评分标准
思考以下三个维度，每个维度给出0-10的分数（整数）。
1. 宏观结构维度
- 故事是否有明确的开端、发展、高潮和结尾？
- 故事的整体结构是否连贯，有无突兀的跳跃或不合理的情节？
- 角色的行动是否有合理的因果关系？
2. 微观结构维度
- 词汇生成是否丰富？（参考词汇总数和不同词汇出现率）
- 句子结构是否复杂？（参考平均句长和句法复杂度）
- 是否使用了修辞手法？
3. 叙事心理描写维度
- 角色的情感表达是否与情节发展一致？
- 是否有角色的心理描写？（对于儿童而言，反映基本情绪反应即可）
4.总分
- 最后，请你权衡各个维度的评分，给出范围0-10的总分。

### 任务数据
1. 这是一名儿童的讲述的故事：
<essay_text>

2. 叙事图
- 从作文中提取了叙事图，叙事图展示了故事中的关键事件及其之间的因果关系，帮助你判断故事宏观结构的组织编排和叙述连贯性,。
- 每个节点代表一个事件，格式为：谓语(主语;宾语;时间状语;地点状语)。
- 边表示事件之间的关系，包括并列、动机-因果、物理-因果、心理-因果和使能-因果。
- 节点序列：{DEFAULT_GRAPH_TOKEN}
- 边结构：<edge_code_str>

3. 微观结构维度评分时，请你使用以下量化数据作为参考：<micro_metrics_str>

### 输出格式
请按照以下格式给出你的回答：
<label_template>

Question：请结合这名儿童讲述的故事和叙事图，告诉我其总分是？让我们一步一步思考。"""


PROMPT_TEMPLATE_v0 = f"""以下是儿童叙事作文的评分标准（1-10分）：
{SCORE_CRITERIA}

现在，给出一篇主题为《青蛙去哪儿了？》的儿童叙事作文：
<essay_text>

叙事图：
- 从作文中提取了叙事图，叙事图展示了故事中的关键事件及其之间的因果关系，帮助你判断故事整体的组织编排和叙述连贯性。
- 每个节点代表一个事件，格式为：谓语(主语;宾语;时间状语;地点状语)。
- 边表示事件之间的关系，包括并列、动机-因果、物理-因果、心理-因果和使能-因果。
- 节点token序列：{DEFAULT_GRAPH_TOKEN}
- 边结构：<edge_code_str>

请根据作文内容和相应的叙事图为该作文评分，并输出以下格式：
预测得分：score"""

PROMPT_TEMPLATE_only_text = f"""你的任务是儿童叙事能力评价，主题是“青蛙，你在哪？”。
以下是儿童叙事能力的评价标准（1-10分）：
{SCORE_CRITERIA}

现在，给出一名儿童讲述的故事：
<essay_text>

请你为该叙事评分，并按以下格式输出结果：
预测得分：score"""


PROMPT_TEMPLATE_TRAIT_ONLY_GRAPH = f"""你的任务是儿童叙事能力评价，主题是“青蛙，你在哪？”。

### 评分标准
思考以下三个维度，每个维度给出0-10的分数（整数）。
1. 宏观结构维度
- 故事是否有明确的开端、发展、高潮和结尾？
- 故事的整体结构是否连贯，有无突兀的跳跃或不合理的情节？
- 角色的行动是否有合理的因果关系？
2. 微观结构维度
- 词汇生成是否丰富？（参考词汇总数和不同词汇出现率）
- 句子结构是否复杂？（参考平均句长和句法复杂度）
- 是否使用了修辞手法？
3. 叙事心理描写维度
- 角色的情感表达是否与情节发展一致？
- 是否有角色的心理描写？（对于儿童而言，反映基本情绪反应即可）
4.总分
- 最后，请你权衡各个维度的评分，给出范围0-10的总分。

### 任务数据

叙事图:
- 从作文中提取了叙事图，叙事图展示了故事中的关键事件及其之间的因果关系，帮助你判断故事宏观结构的组织编排和叙述连贯性,。
- 每个节点代表一个事件，格式为：谓语(主语;宾语;时间状语;地点状语)。
- 边表示事件之间的关系，包括并列、动机-因果、物理-因果、心理-因果和使能-因果。
- 节点序列：{DEFAULT_GRAPH_TOKEN}
- 边结构：<edge_code_str>

### 输出格式
请按照以下格式给出你的回答：
<label_template>

Question：请从宏观结构、微观结构和叙事心理描写维度，评价这名儿童讲述的故事。"""



PROMPT_TEMPLATE_TRAIT_COMMENT = f"""你的任务是儿童叙事能力评价，主题是“青蛙，你在哪？”。

### 评分标准
思考以下三个维度，每个维度给出0-10的分数（整数）。
1. 宏观结构维度
- 故事是否有明确的开端、发展、高潮和结尾？
- 故事的整体结构是否连贯，有无突兀的跳跃或不合理的情节？
- 角色的行动是否有合理的因果关系？
2. 微观结构维度
- 词汇生成是否丰富？（参考词汇总数和不同词汇出现率）
- 句子结构是否复杂？（参考平均句长和句法复杂度）
- 是否使用了修辞手法？
3. 叙事心理描写维度
- 角色的情感表达是否与情节发展一致？
- 是否有角色的心理描写？（对于儿童而言，反映基本情绪反应即可）
4.总分
- 最后，请你权衡各个维度的评分，给出范围0-10的总分。

### 任务数据
1. 这是一名儿童的讲述的故事：
<essay_text>

2. 叙事图
- 从作文中提取了叙事图，叙事图展示了故事中的关键事件及其之间的因果关系，帮助你判断故事宏观结构的组织编排和叙述连贯性,。
- 每个节点代表一个事件，格式为：谓语(主语;宾语;时间状语;地点状语)。
- 边表示事件之间的关系，包括并列、动机-因果、物理-因果、心理-因果和使能-因果。
- 节点序列：{DEFAULT_GRAPH_TOKEN}
- 边结构：<edge_code_str>

3. 微观结构维度评分时，请你使用以下量化数据作为参考：<micro_metrics_str>

### 输出格式
请按照以下格式给出你的回答：
<label_template>

Question：请从宏观结构、微观结构和叙事心理描写维度，评价这名儿童讲述的故事。"""

###########################FEW  SHOT########################################
PROMPT_TEMPLAT_FEWSHOT = f"""你的任务是儿童叙事能力评价，主题是“青蛙，你在哪？”。

### 评分标准
思考以下三个维度，每个维度给出0-10的分数（整数）。
1. 宏观结构维度
- 故事是否有明确的开端、发展、高潮和结尾？
- 故事的整体结构是否连贯，有无突兀的跳跃或不合理的情节？
- 角色的行动是否有合理的因果关系？
2. 微观结构维度
- 词汇生成是否丰富？（参考词汇总数和不同词汇出现率）
- 句子结构是否复杂？（参考平均句长和句法复杂度）
- 是否使用了修辞手法？
3. 叙事心理描写维度
- 角色的情感表达是否与情节发展一致？
- 是否有角色的心理描写？（对于儿童而言，反映基本情绪反应即可）
4.总分
- 最后，请你权衡各个维度的评分，给出范围0-10的总分。

### 示例
1. 第一天,小朋友还有小狗在家里.那个宠物是小青蛙.第二张图是.第一天发生了什么事情呀.第一天的青蛙跳走了.他知道他跳到哪里啦.第三个图是.小朋友太慢了,还打开了窗户.这个小狗掉下去了.他去救了它.他们看见了蜜蜂,还说起了话呀.他看见了蜜蜂.他还.他看见蜜蜂.他还爬了.他看见了蜜蜂.他在爬树.猫头鹰跌倒了.蜜蜂追着小狗跑呀.猫头鹰飞到树上了.这个梅花鹿它救了他,还跑到了这个河上.他们还掉下去了.这张图是.他们掉到水里了.他还看见了这个角上,还找到了树块儿.他们还爬上去了.这个图是.他们看上了一个青蛙.突然他看见了好多青蛙.看见了.他抓了个青蛙.终于可以回家了.
预测得分：1

2. 从前,天黑了.小狗在看青蛙.这天晚上,青蛙出来了.小朋友和小狗在睡觉.第二天,小狗和小朋友发现青蛙不见啦!他在靴子里找.青蛙不在.他在窗户外找.青蛙也不在.小狗要掉下去了.小朋友赶紧跳下去.救到了小狗.小朋友喊道.青蛙,你在哪儿?小朋友要看洞里.蜜蜂来了.老鼠钻出来了.他捂住了鼻子.蜂巢掉了.蜜蜂要去蜇小狗.小朋友赶紧爬到了树上.老鹰飞过来了.小朋友不小心摔倒了.蜜蜂一直追这个小狗.小朋友赶紧捂住头.爬到了石头上去喊.青蛙,你在哪儿?他赶紧骑到了驯鹿上.骑着走.他走到了河边.小朋友和小狗要掉进水里啦.已经掉到水里啦.小朋友和小狗都在水里.他把手捂住了嘴巴和鼻子.上了岸.小狗和小朋友在树头上看到了两只青蛙.赶紧下来了.他救到了青蛙.赶紧划水.最后,就找到了他们的伙伴.
预测得分：6

3. 有一天，我和狗狗发现了一个神奇的玻璃罐子.里面有一只青蛙.虽然说青蛙不是很好看.但是对于我而言能够多一个动物的朋友.我觉得很高兴.我和狗狗仔细地端详着它，充满了好奇.在喂它吃了东西之后.我就是睡觉了.可谁知道一觉醒来.青蛙从瓶子里跑出去了.我和狗狗觉得非常失落，在房间里四处寻找都没有找到.于是我们从窗户爬出去，准备寻找青蛙去哪里了.在爬窗户出去的过程中，狗狗还把装青蛙的瓶子摔碎了.我很生气地看着它.但是为了急着找到青蛙.我们只能先四处呼叫.但是没有得到结果.我们看到一棵大树，觉得青蛙有可能藏在里面，于是前往查看.我看到地上有一个洞，抬头一看是一只臭鼬.狗摇了摇树.从树上掉下来一个马蜂窝.一群马蜂追着它跑.我跑到大树上，朝洞里面看.大树的树洞突然飞出来一只猫头鹰.吓得我从树上掉了下来，但是还是没有找到青蛙.我被猫头鹰追到岩石后面.我还在呼唤.但然后穿出来一只鹿.我把插到地上的角扔到了池塘里面.我和狗狗浑身都湿透了，非常的失落.但是令人惊奇的是.在一个废弃的大树桩上，我看到了两只青蛙.我定睛一看.这不就是我的青蛙朋友嘛.我失落的心情顿时一去不复返.历经千难万险了.我们终于找到青蛙在哪里.我开心地呼唤着青蛙.还剩了一大窝宝宝.看他们幸福地生活在一起.我和狗狗决定不再带它回家，就让它自由地生活.于是我们挥手和青蛙道别.他们也呱呱叫着给我们送别，仿佛在说认识我们很开心.
预测得分：10

### 任务数据
这是一名儿童的讲述的故事：
<essay_text>

### 输出格式
预测得分：<0-10>

Question：这名儿童讲述的故事的总分是？请不要输出额外的解释和维度评分，按照给定的输出格式给出预测的总分。"""

PROMPT_TEMPLATE_TRAIT_COMMENT_FEW_SHOT = f"""你的任务是儿童叙事能力评价，主题是“青蛙，你在哪？”。

### 评分标准
思考以下三个维度，每个维度给出0-10的分数（整数）。
1. 宏观结构维度
- 故事是否有明确的开端、发展、高潮和结尾？
- 故事的整体结构是否连贯，有无突兀的跳跃或不合理的情节？
- 角色的行动是否有合理的因果关系？
2. 微观结构维度
- 词汇生成是否丰富？（参考词汇总数和不同词汇出现率）
- 句子结构是否复杂？（参考平均句长和句法复杂度）
- 是否使用了修辞手法？
3. 叙事心理描写维度
- 角色的情感表达是否与情节发展一致？
- 是否有角色的心理描写？（对于儿童而言，反映基本情绪反应即可）
4.总分
- 最后，请你权衡各个维度的评分，给出范围0-10的总分。

### 示例
1. 第一天,小朋友还有小狗在家里.那个宠物是小青蛙.第二张图是.第一天发生了什么事情呀.第一天的青蛙跳走了.他知道他跳到哪里啦.第三个图是.小朋友太慢了,还打开了窗户.这个小狗掉下去了.他去救了它.他们看见了蜜蜂,还说起了话呀.他看见了蜜蜂.他还.他看见蜜蜂.他还爬了.他看见了蜜蜂.他在爬树.猫头鹰跌倒了.蜜蜂追着小狗跑呀.猫头鹰飞到树上了.这个梅花鹿它救了他,还跑到了这个河上.他们还掉下去了.这张图是.他们掉到水里了.他还看见了这个角上,还找到了树块儿.他们还爬上去了.这个图是.他们看上了一个青蛙.突然他看见了好多青蛙.看见了.他抓了个青蛙.终于可以回家了.
宏观结构得分：2，微观结构得分：3，叙事心理描写得分：0，总分：1。

2. 从前,天黑了.小狗在看青蛙.这天晚上,青蛙出来了.小朋友和小狗在睡觉.第二天,小狗和小朋友发现青蛙不见啦!他在靴子里找.青蛙不在.他在窗户外找.青蛙也不在.小狗要掉下去了.小朋友赶紧跳下去.救到了小狗.小朋友喊道.青蛙,你在哪儿?小朋友要看洞里.蜜蜂来了.老鼠钻出来了.他捂住了鼻子.蜂巢掉了.蜜蜂要去蜇小狗.小朋友赶紧爬到了树上.老鹰飞过来了.小朋友不小心摔倒了.蜜蜂一直追这个小狗.小朋友赶紧捂住头.爬到了石头上去喊.青蛙,你在哪儿?他赶紧骑到了驯鹿上.骑着走.他走到了河边.小朋友和小狗要掉进水里啦.已经掉到水里啦.小朋友和小狗都在水里.他把手捂住了嘴巴和鼻子.上了岸.小狗和小朋友在树头上看到了两只青蛙.赶紧下来了.他救到了青蛙.赶紧划水.最后,就找到了他们的伙伴.
宏观结构得分：6，微观结构得分：6，叙事心理描写得分：5，总分：6。

3. 有一天，我和狗狗发现了一个神奇的玻璃罐子.里面有一只青蛙.虽然说青蛙不是很好看.但是对于我而言能够多一个动物的朋友.我觉得很高兴.我和狗狗仔细地端详着它，充满了好奇.在喂它吃了东西之后.我就是睡觉了.可谁知道一觉醒来.青蛙从瓶子里跑出去了.我和狗狗觉得非常失落，在房间里四处寻找都没有找到.于是我们从窗户爬出去，准备寻找青蛙去哪里了.在爬窗户出去的过程中，狗狗还把装青蛙的瓶子摔碎了.我很生气地看着它.但是为了急着找到青蛙.我们只能先四处呼叫.但是没有得到结果.我们看到一棵大树，觉得青蛙有可能藏在里面，于是前往查看.我看到地上有一个洞，抬头一看是一只臭鼬.狗摇了摇树.从树上掉下来一个马蜂窝.一群马蜂追着它跑.我跑到大树上，朝洞里面看.大树的树洞突然飞出来一只猫头鹰.吓得我从树上掉了下来，但是还是没有找到青蛙.我被猫头鹰追到岩石后面.我还在呼唤.但然后穿出来一只鹿.我把插到地上的角扔到了池塘里面.我和狗狗浑身都湿透了，非常的失落.但是令人惊奇的是.在一个废弃的大树桩上，我看到了两只青蛙.我定睛一看.这不就是我的青蛙朋友嘛.我失落的心情顿时一去不复返.历经千难万险了.我们终于找到青蛙在哪里.我开心地呼唤着青蛙.还剩了一大窝宝宝.看他们幸福地生活在一起.我和狗狗决定不再带它回家，就让它自由地生活.于是我们挥手和青蛙道别.他们也呱呱叫着给我们送别，仿佛在说认识我们很开心.
宏观结构得分：10，微观结构得分：10，叙事心理描写得分：9，总分：10。

### 该儿童讲述的故事
<essay_text>

### 输出格式
宏观结构得分：<macro_score>，微观结构得分：<micro_score>，叙事心理描写得分：<psych_score>，总分：<total_score>。

Question：请从宏观结构、微观结构和叙事心理描写维度，给这名儿童讲述的故事评分。"""


###################### LABEL_TEMPLATE ###############################
LABEL_TEMPLATE_TRAIT_COMMENT = """评语：<comment>
宏观结构得分：<macro_score>，微观结构得分：<micro_score>，叙事心理描写得分：<psych_score>，总分：<total_score>。"""

LABEL_TEMPLATE_TRAIT = """宏观结构得分：<macro_score>，微观结构得分：<micro_score>，叙事心理描写得分：<psych_score>，总分：<total_score>。"""


LABEL_TEMPLATE_SCORE = """预测得分：<0-10>"""

####################### ONLY GRAPH ###########################

PROMPT_TEMPLATE_only_graph = f"""作为一名专业的作文评分员，你的任务是评估儿童在看图叙事任务中所述的叙事文本。

### 写作主题：
《青蛙去哪了？》

### 评分标准：
{SCORE_CRITERIA}

### 叙事图：
叙事图是用于帮助理解事件结构的一个有向图，其中：
- 节点代表一个事件，比如：小男孩爬上树。
- 边表示事件之间的关系，共分为五类。

关系描述： 
{GRAPH_RELATIONS_DESC}

### 叙事图结构：
- 节点序列：<{DEFAULT_GRAPH_TOKEN}>
- 边结构：<edge_code_str>

### 任务：
根据提供的评分标准，为这篇叙事文打分。

### 输出格式：
请勿提供任何额外的评论或解释。请只按以下格式输出分数：
预测分数：<整数分数>

### 输出示例：
预测分数：4  
预测分数：13  

那么，这篇叙事文本的得分是多少？"""

###################################TMP #####################################
SCORE_CRITERIA_TRAIT_COMMENT = """从以下三个维度进行评价，每个维度满分10分，最终根据各维度评分给出总分0-10分（整数）。

#### 1. 宏观结构维度（0-10分）
- 高分（8-10分）：
  - 故事叙述完整，包括主要情节的开端、发展、高潮和结尾，尤其是突出了青蛙逃跑、男孩和狗的冒险和青蛙的回归。
  - 叙述清晰，情节发展合理，逻辑自然，尤其是角色的行动和故事的因果关系（如男孩、狗和蜜蜂的互动，男孩与其他动物的互动等）。
  - 叙述中展示了情节的递进和紧张感，使故事生动有趣。
- 中等分（5-7分）：
  - 叙述结构基本完整，但某些情节略显模糊或仓促，特别是在情节转折或高潮部分。
  - 有时逻辑上稍显不连贯，某些事件的因果关系未能清晰表达。
- 低分（0-4分）：
  - 叙述缺少关键情节，故事不连贯，关键转折点（如青蛙逃跑或结尾）不清晰。
  - 情节之间缺乏逻辑联系，使得故事难以理解。

#### 2. 微观结构维度（0-10分）
- 高分（8-10分）：
  - 词汇生成丰富，不同词汇出现率（TTR）较高，句式多样，平均句长（MLU）较长。
  - 依存树深度大，频繁使用从属连词或关联词，句法复杂。
  - 使用比喻、拟人等修辞手法，增强故事生动性。
- 中等分（5-7分）：
  - 词汇生成适中，TTR一般，句式相对简单，偶有复合句。
  - 依存树深度适中，少量从属连词或关联词。
  - 修辞手法使用较少，表达直白。
- 低分（0-4分）：
  - 词汇生成较少，TTR低，句式单一，几乎没有使用复杂句。
  - 依存树深度低，缺少从属连词或关联词的使用。
  - 修辞手法缺乏，表达平淡。

#### 3. 叙事心理描写维度（0-10分）
- 高分（8-10分）：
  - 角色情感描写细腻，自然反映不同情境下的情绪（如焦急、兴奋、失望），与情节紧密结合。
  - 心理描写推动情节发展，增强读者的代入感。
- 中等分（5-7分）：
  - 情感描写基本到位，情绪表达清楚，但细节不足，情感起伏不明显。
- 低分（0-4分）（低年龄儿童心理维度要求降低）：
  - 情感描写较弱或缺失，情感表达不自然或生硬。
  - 低年龄儿童在此维度中不会被严格要求过多细腻的情感描写，但仍需体现基本的情绪反应。"""



example="""示例：
评语：宏观叙事方面缺少部分在外找青蛙和主人公遭遇的情节，使这部分不够连贯。语句通顺词汇丰富，但句子结构较单一，且心理状态描写较少。
宏观结构得分：7，微观结构得分：8，叙事心理描写得分：3，总分：7。"""





LABEL_TEMPLATE_V2 = """评语：<comment>
- 宏观结构得分：<macro_score>
- 微观结构得分：<micro_score>
- 叙事心理描写得分：<psych_score>
- 总分：<total_score>"""

PROMPT_TEMPLATE_TRAIT_COMMENT_V2 = f"""你的任务是儿童叙事能力评价，主题是“青蛙，你在哪？”。

### 评分标准：
{SCORE_CRITERIA_TRAIT_COMMENT}

### 该儿童的叙事文：
<essay_text>

### 叙事图：
- 叙事图展示了故事中的关键事件及其之间的因果关系。每个节点代表一个事件，格式为：谓语(主语;宾语;时间状语;地点状语)。
- 边表示事件之间的关系，包括并列、动机-因果、物理-因果、心理-因果和使能-因果。格式为：(起始事件id,目标事件id,事件关系)

- 节点序列：{DEFAULT_GRAPH_TOKEN}
- 边结构：<edge_code_str>


### 提示：
1. 在宏观结构维度评估时，充分利用叙事图分析事件的关键节点、因果关系及连贯性，以判断故事整体的组织编排和叙述连贯性。
2. 评估顺序为：宏观结构 → 微观结构 → 叙事心理描写 → 总分，确保每个维度逐步分析，全面评估叙事能力。


### 输出格式：
评语：
- 宏观结构得分：
- 微观结构得分：
- 叙事心理描写得分：
- 总分：

请根据叙事文本、叙事图和评分标准，严格按照给定的格式，给出详细的评语、各维度的评分及总分。"""






example = """### 评分示例：
叙事文：
一天晚上，我和我的小狗在房间里面观察瓶子里面的青蛙.看着看着我累了我就上床睡觉了小狗也在我的床上和我一起睡觉在我们睡觉的时候，那只青蛙从玻璃瓶里面跑了出去。第二天早上，我们醒来之后发现青蛙消失了。我们都很着急然后我跟小狗就在房间里面找来找去我翻了这个靴子小狗把头伸进了玻璃瓶都没找到。这时，我和小狗趴在窗台上面戴着瓶子的小狗一不小心掉了下去，还把玻璃瓶打碎了。我批评了小狗后来我们俩一起出去去找小青蛙我们大叫，但是也没有找到它后来我们来到了一棵树下.树上面挂着一个蜂窝.小狗在那儿叫.把这些蜜蜂都引了出来.我看到一个洞.突然，跑出来了一只仓鼠.后来，我爬上了树枝小狗在那儿把蜂窝摇了下来。于是，那个蜂窝里面的蜜蜂就追着小狗跑。我被这一群蜂蜜给吓倒了还出现了一只老鹰老鹰在我爬上石头的时候，往我头上飞过我爬上石头，在那儿呼唤小青蛙没想到我抓住的那个枝干是鹿的犄角后来鹿把我带到了悬崖边。但没想到我们掉了下去掉下去之后呢，下面是一个池塘。然后小狗就趴在我的头上在池塘旁边有一个枯树干我和小狗，我示意让小狗悄悄的于是呢我和小狗趴在那儿观察树干后面的东西，发现小青蛙跟他妈妈在一起后来，我们将小青蛙带走了跟青蛙妈妈和她的宝宝们道别了

评语：宏观叙事方面缺少部分在外找青蛙和主人公遭遇的情节，使这部分不够连贯。语句通顺词汇丰富，但句子结构较单一，且心理状态描写较少。
- 宏观结构得分：7
- 微观叙事得分：8
- 叙事心理描写得分：3
- 总分：7"""


















'''你的任务是儿童叙事能力评价，主题是“青蛙，你在哪？”。

### 评分标准
思考以下三个维度，每个维度给出1-10的分数（整数）。
1. 宏观结构维度
- 故事是否有明确的开端、发展、高潮和结尾？
- 故事的整体结构是否连贯，有无突兀的跳跃或不合理的情节？
- 角色的行动是否有合理的因果关系？
2. 微观结构维度
- 词汇生成是否丰富？（参考词汇总数和不同词汇出现率）
- 句子结构是否复杂？（参考平均句长和句法复杂度）
- 是否使用了修辞手法？
3. 叙事心理描写维度
- 角色的情感表达是否与情节发展一致？
- 是否有角色的心理描写？（对于儿童而言，反映基本情绪反应即可）
4.总分
- 最后，请你权衡各个维度的评分，给出范围1-10的总分。

### 任务数据
叙事图
- 从作文中提取了叙事图，叙事图展示了故事中的关键事件及其之间的因果关系，帮助你判断故事宏观结构的组织编排和叙述连贯性,。
- 每个节点代表一个事件，格式为：谓语(主语;宾语;时间状语;地点状语)。
- 边表示事件之间的关系，包括并列、动机-因果、物理-因果、心理-因果和使能-因果。

Question：请从宏观结构、微观结构和叙事心理描写维度，给出这名儿童讲述的故事的分数。'''










#Score 0: The narrative is entirely incoherent, with no discernible structure or meaningful content. There are no clear events, and any attempt at causal relationships is nonexistent. Vocabulary use is extremely poor, with many repeated words and almost no descriptive language. Sentence structure is either too simplistic or completely fragmented, making comprehension impossible. 
#Score 1: The narrative includes very few events, which are poorly structured and difficult to understand. The causal relationships are extremely vague, with almost no connection between events. The vocabulary is very limited, lacking descriptive and varied language. Sentence structure is very simple, frequently containing grammatical errors that hinder comprehension. 
SCORE_CRITERIA_en = """Score 2: The narrative presents some events, but they are not clearly described and lack a complete logical structure. Causal relationships are unclear and weakly established. Vocabulary use is limited, with some repetitive words and a lack of vividness. Sentence structure is mostly simple, with occasional grammatical mistakes. 
Score 3: Events in the narrative begin to emerge, but the description is still incomplete and somewhat disjointed. Causal relationships occasionally appear but are not consistently maintained. The vocabulary shows some increase, including a few descriptive words, but remains limited. Sentence structure is basic, with simple sentences predominating and occasional grammatical errors. 
Score 4: The narrative outlines basic events clearly, but lacks detailed expansion. Causal relationships are generally present but sometimes unclear. Vocabulary usage is moderate, with some descriptive language but not extensive. Sentence structure is relatively simple, with some variety, and fewer grammatical errors. 
Score 5: The narrative provides a complete description of events, with a basic beginning, middle, and end. Causal relationships are generally clear but may occasionally lack logical rigor. Vocabulary is reasonably rich, showing some diversity and descriptive quality. Sentence structure is varied, including a mix of simple and compound sentences. 
Score 6: The narrative describes events in detail, with a clear and coherent plot. Causal relationships are well-established and logical. Vocabulary is quite rich, featuring diverse and descriptive language. Sentence structure is diverse and includes some complexity, with the overall flow being smooth. 
Score 7: The narrative offers detailed descriptions of events, with a clear development arc. Causal relationships are clearly and consistently maintained, showing tight logic. Vocabulary is rich and expressive, with considerable diversity. Sentence structure is varied and complex, showing a good command of syntax. 
Score 8: The narrative is very well-detailed, with a vivid and coherent storyline. Causal relationships are extremely clear and logically tight, with natural transitions. Vocabulary is highly rich and precise, with strong descriptive quality. Sentence structure is complex and varied, with accurate grammar throughout. 
Score 9: The narrative provides extremely detailed and rich descriptions of events, with a highly engaging plot. Causal relationships are very clear and logical, with a strong sense of narrative consistency. Vocabulary is exceptionally rich and layered, offering vivid and precise expression. Sentence structure is highly complex and polished, with flawless grammar. 
Score 10: The narrative is exceptionally well-crafted, offering a vivid and intricate description of events with meticulous attention to detail. Causal relationships are perfectly clear and seamlessly integrated, with impeccable logic and narrative flow. Vocabulary is extraordinarily rich, nuanced, and literary. Sentence structure is highly sophisticated and elegant, with perfect grammar.
"""

PROMPT_TEMPLATE_V1 = f"""The criteria for scoring this children's narrative essay (2-10) are: 
{SCORE_CRITERIA}

Now, here is a children's narrative essay with the theme "Where did the frog go?": 
<essay_text>

A narrative graph has been extracted from the essay (directed graph；nodes represent events; edges represent relationships between events such as enable cause, motivation cause, physical cause, psychological cause, and parallel relationships). 
The node sequence of the narrative graph is: 
{DEFAULT_GRAPH_TOKEN}
The structure of edges is described by: 
<edge_code_str>

Please provide a score for this essay based on both the narrative text and the corresponding graph, and output the following format: 
Prediction score: <score>."""
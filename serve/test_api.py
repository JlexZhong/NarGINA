import json
import requests

# 定义 Flask 服务的 URL
url = 'http://localhost:5000/narrative'

story = ['从前有一个小男孩他养了一只青蛙.',
'有一天晚上小男孩睡觉的时候青蛙偷偷地爬出来了.',
'然后小男孩起来的时候就找不到青蛙.',
'他自己起来.',
'让小狗套了一个瓶子.',
'然后自己弄弄了一个靴子.',
'窗户旁边找小青蛙.',
'然后呢小狗跳下去了.',
'然后那个小男孩就突然一下下去抱住了小狗.',
'他走到一个森林旁边喊了一句小青蛙.',
'可是没有小青蛙的声音.',
'他发现有一个洞.',
'洞旁边有一棵树.',
'树上有一个窝.',
'窝里面不知道是什么.',
'他就往洞里喊小青蛙你在这里吗?',
'突然从这个洞里跑出来一个小老鼠说我不是小青蛙我是小老鼠.',
'小狗这时看见那棵树上的那个窝里面有很多蜜蜂飞出来.',
'他就嗯很惊讶.',
'他就一不小心把那个东西弄翻了.',
'有很多蜜蜂都跑出来了.',
'小男孩发现一个大树大树洞里面可能就是小青蛙吧.',
'小男孩往里面喊小青蛙你在这里吗?',
'突然有一只猫头鹰飞过来.',
'附近没有什么小青蛙我是猫头鹰!',
'猫头鹰老追着他说没有什么小青蛙我是猫头鹰.',
'小男孩喊一边找小青蛙一边喊着说小青蛙你在哪里?',
'他掉到一个树干上.',
'唉？这是什么东西?',
'哦原来是一个鹿.',
'鹿老是在这片丛林里路过.',
'然后呢鹿就把他弄到自己身上去了.',
'然后小男孩就非得让那个鹿去背他.',
'那个鹿看见河水然后呢就把小男孩给给摔下去了.',
'小狗也摔下去了.',
'然后他们就游啊.',
'没看见什么东西.',
'嗯这个鹿在喝水的时候没看见这边是一个悬崖就喝不到水.',
'小男孩发现了一个树棍.',
'他就爬上去.',
'然后他在上面休息休息.',
'他听到呱呱呱的声音.',
'他就叫小狗说好像听到青蛙的声音了.',
'他们翻过去一看真是青蛙呀.',
'所有小青蛙和两只大青蛙在那边.',
'他们这时带走了一只小青蛙.',
'高高兴兴地回到家里去了.'
]

# 假设你要传递一些 JSON 数据（如果需要的话）
data = {
    'story':story
}

# 发起 POST 请求
response = requests.post(url, json=data)

# 检查响应状态
if response.status_code == 200:
    # 如果请求成功，处理返回的数据
    content = response.json()  # 获取返回的 JSON 数据
    with open("content.json", "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    print("Response:", content)
else:
    print("Error:", response.json())

url = "http://127.0.0.1:5000/get_image"

response = requests.get(url)

if response.status_code == 200:
    with open("downloaded_image.png", "wb") as f:
        f.write(response.content)
    print("Image downloaded successfully as downloaded_image.png")
else:
    print(f"Failed to get image. Status code: {response.status_code}")
    print(response.json())
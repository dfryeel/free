from graphviz import Digraph

def make_res2net_graph():
    dot = Digraph(comment='Res2Net')

    # 输入节点
    dot.node('input', 'Input (3, H, W)')

    # 初始卷积层
    dot.node('conv1', 'Conv1 (32, H/2, W/2)')
    dot.edge('input', 'conv1')

    dot.node('bn1', 'BN1 (32, H/2, W/2)')
    dot.edge('conv1', 'bn1')

    dot.node('relu1', 'ReLU1 (32, H/2, W/2)')
    dot.edge('bn1', 'relu1')

    dot.node('conv2', 'Conv2 (32, H/2, W/2)')
    dot.edge('relu1', 'conv2')

    dot.node('bn2', 'BN2 (32, H/2, W/2)')
    dot.edge('conv2', 'bn2')

    dot.node('relu2', 'ReLU2 (32, H/2, W/2)')
    dot.edge('bn2', 'relu2')

    dot.node('conv3', 'Conv3 (64, H/2, W/2)')
    dot.edge('relu2', 'conv3')

    dot.node('bn3', 'BN3 (64, H/2, W/2)')
    dot.edge('conv3', 'bn3')

    dot.node('relu3', 'ReLU3 (64, H/2, W/2)')
    dot.edge('bn3', 'relu3')

    dot.node('maxpool', 'MaxPool (64, H/4, W/4)')
    dot.edge('relu3', 'maxpool')

    # 主要残差块层
    dot.node('layer1', 'Layer1 (64, H/4, W/4)')
    dot.edge('maxpool', 'layer1')

    dot.node('layer2', 'Layer2 (128, H/8, W/8)')
    dot.edge('layer1', 'layer2')

    dot.node('layer3', 'Layer3 (256, H/16, W/16)')
    dot.edge('layer2', 'layer3')

    dot.node('layer4', 'Layer4 (512, H/32, W/32)')
    dot.edge('layer3', 'layer4')

    # 输出特征图
    dot.node('output1', 'Output1 (64, H/4, W/4)')
    dot.edge('layer1', 'output1')

    dot.node('output2', 'Output2 (128, H/8, W/8)')
    dot.edge('layer2', 'output2')

    dot.node('output3', 'Output3 (256, H/16, W/16)')
    dot.edge('layer3', 'output3')

    dot.node('output4', 'Output4 (512, H/32, W/32)')
    dot.edge('layer4', 'output4')

    return dot

# 生成并保存图形
dot = make_res2net_graph()
dot.render('res2net_structure', view=True)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from utils.dataImport import load_clean_dataset
from utils.decisionTree import decision_tree_learning


def _generate_lines(node, x=10, y=0, length=100, fontSize=10):
    """
    Generate lines for the decision tree visualiser
    """
    segments = []
    label = "X" + str(node['attr']) + "<" + str(
        node["value"]) if not node['isLeaf'] else "LEAF:" + str(node['label'])
    plt.annotate(label, (x, y), fontSize=max(fontSize, 5))
    if 'left' in node:
        newX = x + (50 if not node['left']['isLeaf'] else 25)
        newY = y + length
        seg1, seg2 = [[x, y], [x, newY]], \
                     [[x, newY], [newX, newY]]
        subSegs = _generate_lines(node['left'], newX, newY, length * 0.5, fontSize - 1)
        segments.extend([seg1, seg2])
        segments.extend(subSegs)
    if 'right' in node:
        newX = x + (50 if not node['right']['isLeaf'] else 25)
        newY = y - length
        seg1, seg2 = [[x, y], [x, newY]], \
                     [[x, newY], [newX, newY]]
        subSegs = _generate_lines(node['right'], newX, newY, length * 0.5, fontSize - 1)
        segments.extend([seg1, seg2])
        segments.extend(subSegs)
    return segments


def show_tree(node, maxDepth):
    """
    Show a tree visualiser, given the tree and it's maximum depth
    """
    fig, ax = plt.subplots()
    xMax = maxDepth * 60
    yMax = maxDepth * maxDepth * 2 + 50
    ax.set_xlim(0, xMax)
    ax.set_ylim(yMax, 3)
    lines = _generate_lines(node, y=yMax / 2)
    ax.add_collection(LineCollection(lines))
    fig.subplots_adjust(bottom=0.2)
    plt.show()


def show_entire_clean_tree():
    """
    Show the decision tree for the clean dataset
    """
    clean_dataset = load_clean_dataset()
    tree, depth = decision_tree_learning(clean_dataset)
    show_tree(tree, depth)

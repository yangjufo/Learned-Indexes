# BTree Index with Python

import pandas as pd

# Node in BTree
class BTreeNode:
    def __init__(self, degree=2, number_of_keys=0, is_leaf=True, items=None, children=None,
                 index=None):
        self.isLeaf = is_leaf
        self.numberOfKeys = number_of_keys
        self.index = index
        if items is not None:
            self.items = items
        else:
            self.items = [None] * (degree * 2 - 1)
        if children is not None:
            self.children = children
        else:
            self.children = [None] * degree * 2

    def set_index(self, index):
        self.index = index

    def get_index(self):
        return self.index

    def search(self, b_tree, an_item):
        i = 0
        while i < self.numberOfKeys and an_item > self.items[i]:
            i += 1
        if i < self.numberOfKeys and an_item == self.items[i]:
            return {'found': True, 'fileIndex': self.index, 'nodeIndex': i}
        if self.isLeaf:
            return {'found': False, 'fileIndex': self.index, 'nodeIndex': i - 1}
        else:
            return b_tree.get_node(self.children[i]).search(b_tree, an_item)

# BTree Class
class BTree:
    def __init__(self, degree=2, nodes=None, root_index=1, free_index=2):
        if nodes is None:
            nodes = {}
        self.degree = degree
        if len(nodes) == 0:
            self.rootNode = BTreeNode(degree)
            self.nodes = {}
            self.rootNode.set_index(root_index)
            self.write_at(1, self.rootNode)
        else:
            self.nodes = nodes
            self.rootNode = self.nodes[root_index]
        self.rootIndex = root_index
        self.freeIndex = free_index

    def build(self, keys, values):
        if len(keys) != len(values):
            return
        for ind in range(len(keys)):
            self.insert(Item(keys[ind], values[ind]))

    def search(self, an_item):
        return self.rootNode.search(self, an_item)

    def predict(self, key):
        search_result = self.search(Item(key, 0))
        a_node = self.nodes[search_result['fileIndex']]
        if a_node.items[search_result['nodeIndex']] is None:
            return -1
        return a_node.items[search_result['nodeIndex']].v

    def split_child(self, p_node, i, c_node):
        new_node = self.get_free_node()
        new_node.isLeaf = c_node.isLeaf
        new_node.numberOfKeys = self.degree - 1
        for j in range(0, self.degree - 1):
            new_node.items[j] = c_node.items[j + self.degree]
        if c_node.isLeaf is False:
            for j in range(0, self.degree):
                new_node.children[j] = c_node.children[j + self.degree]
        c_node.numberOfKeys = self.degree - 1
        j = p_node.numberOfKeys + 1
        while j > i + 1:
            p_node.children[j + 1] = p_node.children[j]
            j -= 1
        p_node.children[j] = new_node.get_index()
        j = p_node.numberOfKeys
        while j > i:
            p_node.items[j + 1] = p_node.items[j]
            j -= 1
        p_node.items[i] = c_node.items[self.degree - 1]
        p_node.numberOfKeys += 1

    def insert(self, an_item):
        search_result = self.search(an_item)
        if search_result['found']:
            return None
        r = self.rootNode
        if r.numberOfKeys == 2 * self.degree - 1:
            s = self.get_free_node()
            self.set_root_node(s)
            s.isLeaf = False
            s.numberOfKeys = 0
            s.children[0] = r.get_index()
            self.split_child(s, 0, r)
            self.insert_not_full(s, an_item)
        else:
            self.insert_not_full(r, an_item)

    def insert_not_full(self, inNode, anItem):
        i = inNode.numberOfKeys - 1
        if inNode.isLeaf:
            while i >= 0 and anItem < inNode.items[i]:
                inNode.items[i + 1] = inNode.items[i]
                i -= 1
            inNode.items[i + 1] = anItem
            inNode.numberOfKeys += 1
        else:
            while i >= 0 and anItem < inNode.items[i]:
                i -= 1
            i += 1
            if self.get_node(inNode.children[i]).numberOfKeys == 2 * self.degree - 1:
                self.split_child(inNode, i, self.get_node(inNode.children[i]))
                if anItem > inNode.items[i]:
                    i += 1
            self.insert_not_full(self.get_node(inNode.children[i]), anItem)

    def delete(self, an_item):
        an_item = Item(an_item, 0)
        search_result = self.search(an_item)
        if search_result['found'] is False:
            return None
        r = self.rootNode
        self.delete_in_node(r, an_item, search_result)

    def delete_in_node(self, a_node, an_item, search_result):
        if a_node.index == search_result['fileIndex']:
            i = search_result['nodeIndex']
            if a_node.isLeaf:
                while i < a_node.numberOfKeys - 1:
                    a_node.items[i] = a_node.items[i + 1]
                    i += 1
                a_node.numberOfKeys -= 1
            else:
                left = self.get_node(a_node.children[i])
                right = self.get_node(a_node.children[i + 1])
                if left.numberOfKeys >= self.degree:
                    a_node.items[i] = self.get_right_most(left)
                elif right.numberOfKeys >= self.degree:
                    a_node.items[i] = self.get_right_most(right)
                else:
                    k = left.numberOfKeys
                    left.items[left.numberOfKeys] = an_item
                    left.numberOfKeys += 1
                    for j in range(0, right.numberOfKeys):
                        left.items[left.numberOfKeys] = right.items[j]
                        left.numberOfKeys += 1
                    del self.nodes[right.get_index()]
                    for j in range(i, a_node.numberOfKeys - 1):
                        a_node.items[j] = a_node.items[j + 1]
                        a_node.children[j + 1] = a_node.children[j + 2]
                    a_node.numberOfKeys -= 1
                    if a_node.numberOfKeys == 0:
                        del self.nodes[a_node.get_index()]
                        self.set_root_node(left)
                    self.delete_in_node(left, an_item, {'found': True, 'fileIndex': left.index, 'nodeIndex': k})
        else:
            i = 0
            while i < a_node.numberOfKeys and self.get_node(a_node.children[i]).search(self, an_item)['found'] is False:
                i += 1
            c_node = self.get_node(a_node.children[i])
            if c_node.numberOfKeys < self.degree:
                j = i - 1
                while j < i + 2 and self.get_node(a_node.children[j]).numberOfKeys < self.degree:
                    j += 1
                if j == i - 1:
                    sNode = self.get_node(a_node.children[j])
                    k = c_node.numberOfKeys
                    while k > 0:
                        c_node.items[k] = c_node.items[k - 1]
                        c_node.children[k + 1] = c_node.children[k]
                        k -= 1
                    c_node.children[1] = c_node.children[0]
                    c_node.items[0] = a_node.items[i - 1]
                    c_node.children[0] = sNode.children[sNode.numberOfKeys]
                    c_node.numberOfKeys += 1
                    a_node.items[i - 1] = sNode.items[sNode.numberOfKeys - 1]
                    sNode.numberOfKeys -= 1
                elif j == i + 1:
                    sNode = self.get_node(a_node.children[j])
                    c_node.items[c_node.numberOfKeys] = a_node.items[i]
                    c_node.children[c_node.numberOfKeys + 1] = sNode.children[0]
                    a_node.items[i] = sNode.items[0]
                    for k in range(0, sNode.numberOfKeys):
                        sNode.items[k] = sNode.items[k + 1]
                        sNode.children[k] = sNode.children[k + 1]
                    sNode.children[k] = sNode.children[k + 1]
                    sNode.numberOfKeys -= 1
                else:
                    j = i + 1
                    sNode = self.get_node(a_node.children[j])
                    c_node.items[c_node.numberOfKeys] = a_node.items[i]
                    c_node.numberOfKeys += 1
                    for k in range(0, sNode.numberOfKeys):
                        c_node.items[c_node.numberOfKeys] = sNode.items[k]
                        c_node.numberOfKeys += 1
                    del self.nodes[sNode.index]
                    for k in range(i, a_node.numberOfKeys - 1):
                        a_node.items[i] = a_node.items[i + 1]
                        a_node.children[i + 1] = a_node.items[i + 2]
                    a_node.numberOfKeys -= 1
                    if a_node.numberOfKeys == 0:
                        del self.nodes[a_node.index]
                        self.set_root_node(c_node)
            self.delete_in_node(c_node, an_item, c_node.search(self, an_item))

    def get_right_most(self, aNode):
        if aNode.children[aNode.numberOfKeys] is None:
            upItem = aNode.items[aNode.numberOfKeys - 1]
            self.delete_in_node(aNode, upItem,
                                {'found': True, 'fileIndex': aNode.index, 'nodeIndex': aNode.numberOfKeys - 1})
            return upItem
        else:
            return self.get_right_most(self.get_node(aNode.children[aNode.numberOfKeys]))

    def set_root_node(self, r):
        self.rootNode = r
        self.rootIndex = self.rootNode.get_index()

    def get_node(self, index):
        return self.nodes[index]

    def get_free_node(self):
        new_node = BTreeNode(self.degree)
        index = self.get_free_index()
        new_node.set_index(index)
        self.write_at(index, new_node)
        return new_node

    def get_free_index(self):
        self.freeIndex += 1
        return self.freeIndex - 1

    def write_at(self, index, a_node):
        self.nodes[index] = a_node

# Value in Node
class Item():
    def __init__(self, k, v):
        self.k = k
        self.v = v

    def __gt__(self, other):
        if self.k > other.k:
            return True
        else:
            return False

    def __ge__(self, other):
        if self.k >= other.k:
            return True
        else:
            return False

    def __eq__(self, other):
        if self.k == other.k:
            return True
        else:
            return False

    def __le__(self, other):
        if self.k <= other.k:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.k < other.k:
            return True
        else:
            return False

# For Test
def b_tree_main():
    path = "last_data.csv"
    data = pd.read_csv(path)
    b = BTree(2)
    for i in range(data.shape[0]):
        b.insert(Item(data.ix[i, 0], data.ix[i, 1]))

    pos = b.predict(30310)
    print(pos)


if __name__ == '__main__':
    b_tree_main()

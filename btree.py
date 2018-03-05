class BTreeNode:
    def __init__(self, degree=2, numberOfKeys=0, isLeaf=True, items=None, children=None,
                 index=None):
        self.isLeaf = isLeaf
        self.numberOfKeys = numberOfKeys
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

    def search(self, bTree, anItem):
        i = 0
        while i < self.numberOfKeys and anItem > self.items[i]:
            i += 1
        if i < self.numberOfKeys and anItem == self.items[i]:
            return {'found': True, 'fileIndex': self.index, 'nodeIndex': i}
        if self.isLeaf:
            return {'found': False, 'fileIndex': None, 'nodeIndex': None}
        else:
            return bTree.get_node(self.children[i]).search(bTree, anItem)


class BTree:
    def __init__(self, degree=2, nodes={}, rootIndex=1, freeIndex=2):
        self.degree = degree
        if len(nodes) == 0:
            self.rootNode = BTreeNode(degree)
            self.nodes = {}
            self.rootNode.set_index(rootIndex)
            self.write_at(1, self.rootNode)
        else:
            self.nodes = nodes
            self.rootNode = self.nodes[rootIndex]
        self.rootIndex = rootIndex
        self.freeIndex = freeIndex

    def search(self, anItem):
        return self.rootNode.search(self, anItem)

    def split_child(self, pNode, i, cNode):
        newNode = self.get_free_node()
        newNode.isLeaf = cNode.isLeaf
        newNode.numberOfKeys = self.degree - 1
        for j in range(0, self.degree - 1):
            newNode.items[j] = cNode.items[j + self.degree]
        if cNode.isLeaf is False:
            for j in range(0, self.degree):
                newNode.children[j] = cNode.children[j + self.degree]
        cNode.numberOfKeys = self.degree - 1
        j = pNode.numberOfKeys + 1
        while j > i + 1:
            pNode.children[j + 1] = pNode.children[j]
            j -= 1
        pNode.children[j] = newNode.get_index()
        j = pNode.numberOfKeys
        while j > i:
            pNode.items[j + 1] = pNode.items[j]
            j -= 1
        pNode.items[i] = cNode.items[self.degree - 1]
        pNode.numberOfKeys += 1

    def insert(self, anItem):
        searchResult = self.search(anItem)
        if searchResult['found']:
            return None
        r = self.rootNode
        if r.numberOfKeys == 2 * self.degree - 1:
            s = self.get_free_node()
            self.set_root_node(s)
            s.isLeaf = False
            s.numberOfKeys = 0
            s.children[0] = r.get_index()
            self.split_child(s, 0, r)
            self.insert_not_full(s, anItem)
        else:
            self.insert_not_full(r, anItem)

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

    def delete(self, anItem):
        searchResult = self.search(anItem)
        if searchResult['found'] is False:
            return None
        r = self.rootNode
        self.delete_in_node(r, anItem, searchResult)

    def delete_in_node(self, aNode, anItem, searchResult):
        if aNode.index == searchResult['fileIndex']:
            i = searchResult['nodeIndex']
            if aNode.isLeaf:
                while i < aNode.numberOfKeys - 1:
                    aNode.items[i] = aNode.items[i + 1]
                    i += 1
                aNode.numberOfKeys -= 1
            else:
                left = self.get_node(aNode.children[i])
                right = self.get_node(aNode.children[i + 1])
                if left.numberOfKeys >= self.degree:
                    aNode.items[i] = self.get_right_most(left)
                elif right.numberOfKeys >= self.degree:
                    aNode.items[i] = self.get_right_most(right)
                else:
                    k = left.numberOfKeys
                    left.items[left.numberOfKeys] = anItem
                    left.numberOfKeys += 1
                    for j in range(0, right.numberOfKeys):
                        left.items[left.numberOfKeys] = right.items[j]
                        left.numberOfKeys += 1
                    del self.nodes[right.get_index()]
                    for j in range(i, aNode.numberOfKeys - 1):
                        aNode.items[j] = aNode.items[j + 1]
                        aNode.children[j + 1] = aNode.children[j + 2]
                    aNode.numberOfKeys -= 1
                    if aNode.numberOfKeys == 0:
                        del self.nodes[aNode.get_index()]
                        self.set_root_node(left)
                    self.delete_in_node(left, anItem, {'found': True, 'fileIndex': left.index, 'nodeIndex': k})
        else:
            i = 0
            while i < aNode.numberOfKeys and self.get_node(aNode.children[i]).search(self, anItem)['found'] is False:
                i += 1
            cNode = self.get_node(aNode.children[i])
            if cNode.numberOfKeys < self.degree:
                j = i - 1
                while j < i + 2 and self.get_node(aNode.children[j]).numberOfKeys < self.degree:
                    j += 1
                if j == i - 1:
                    sNode = self.get_node(aNode.children[j])
                    k = cNode.numberOfKeys
                    while k > 0:
                        cNode.items[k] = cNode.items[k - 1]
                        cNode.children[k + 1] = cNode.children[k]
                        k -= 1
                    cNode.children[1] = cNode.children[0]
                    cNode.items[0] = aNode.items[i - 1]
                    cNode.children[0] = sNode.children[sNode.numberOfKeys]
                    cNode.numberOfKeys += 1
                    aNode.items[i - 1] = sNode.items[sNode.numberOfKeys - 1]
                    sNode.numberOfKeys -= 1
                elif j == i + 1:
                    sNode = self.get_node(aNode.children[j])
                    cNode.items[cNode.numberOfKeys] = aNode.items[i]
                    cNode.children[cNode.numberOfKeys + 1] = sNode.children[0]
                    aNode.items[i] = sNode.items[0]
                    for k in range(0, sNode.numberOfKeys):
                        sNode.items[k] = sNode.items[k + 1]
                        sNode.children[k] = sNode.children[k + 1]
                    sNode.children[k] = sNode.children[k + 1]
                    sNode.numberOfKeys -= 1
                else:
                    j = i + 1
                    sNode = self.get_node(aNode.children[j])
                    cNode.items[cNode.numberOfKeys] = aNode.items[i]
                    cNode.numberOfKeys += 1
                    for k in range(0, sNode.numberOfKeys):
                        cNode.items[cNode.numberOfKeys] = sNode.items[k]
                        cNode.numberOfKeys += 1
                    del self.nodes[sNode.index]
                    for k in range(i, aNode.numberOfKeys - 1):
                        aNode.items[i] = aNode.items[i + 1]
                        aNode.children[i + 1] = aNode.items[i + 2]
                    aNode.numberOfKeys -= 1
                    if aNode.numberOfKeys == 0:
                        del self.nodes[aNode.index]
                        self.set_root_node(cNode)
            self.delete_in_node(cNode, anItem, cNode.search(self, anItem))

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
        newNode = BTreeNode(self.degree)
        index = self.get_free_index()
        newNode.set_index(index)
        self.write_at(index, newNode)
        return newNode

    def get_free_index(self):
        self.freeIndex += 1
        return self.freeIndex - 1

    def write_at(self, index, aNode):
        self.nodes[index] = aNode


def b_tree_main():
    lst = [10, 8, 22, 14, 12, 18, 2, 50, 15]

    b = BTree(2)

    for x in lst:
        print("***Inserting", x)
        b.insert(x)

    for x in lst:
        print("***Deleting", x)
        b.delete(x)

    b.insert(10)


if __name__ == '__main__':
    b_tree_main()

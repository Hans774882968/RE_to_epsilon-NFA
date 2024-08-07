[TOC]

## 引言

库：

- numpy：手算posG
- matplotlib：可视化
- networkx：可视化

以下命令可以从豆瓣源下networkx：`pip install -i https://pypi.doubanio.com/simple networkx`

## 一、解析RE

RE本身是上下文无关文法（俄罗斯套娃么qwq……CFG本身作为字符串是什么语言呢？），写出文法（注意规避左递归等）：

```c++
S -> T + S
T -> E . T
E -> termR | (S)R//term是一个终结符
R -> 0到多个"*"
```

之前想过用一个栈模拟PDA来实现，但是还是屈服了因为我太鶸了QAQ。所以还是递归实现（原理：模拟parse tree生成）：

```python
i = 0
s = ""
token = ""
g = None  # 手打class MyGraph的实例

# 向前读一个字符（可以用yield代替）
def advance():
    global i, s, token
    if i >= len(s):
        token = None
    else:
        token = s[i]
        i += 1
    return token


def R(st, ed):
    global i, s, token, g
    advance()
    while token == "*":
        st, ed = g.addedge_star(st, ed)
        advance()
    return st, ed


def E():
    global i, s, token, g
    ch = advance()
    if ch and (ch.islower() or ch == "ε" or ch.isdigit()):
        st, ed = g.addedge_base(ch)
    elif ch and ch == "(":
        fl, st, ed = parse_RE()
        if not fl or token != ")":
            return False, 0, 0
    else:
        return False, 0, 0
    st, ed = R(st, ed)
    return True, st, ed


def T():
    global i, s, token, g
    fl1, st1, ed1 = E()
    fl2 = True
    if token == '.':
        fl2, st2, ed2 = T()
        st1, ed1 = g.addedge_concat(st1, ed1, st2, ed2)
    return (fl1 and fl2), st1, ed1


def parse_RE():
    global i, s, token, g
    fl1, st1, ed1 = T()
    st, ed = [st1], [ed1]
    fl2 = True
    while token == "+":
        fl2, st2, ed2 = T()
        fl1 = fl1 and fl2
        if not fl1:
            return fl1, 0, 0
        st.append(st2)
        ed.append(ed2)
    if len(st) > 1:
        st1, ed1 = g.addedge_union(st, ed)
    return fl1, st1, ed1
```

生成parse tree的过程可以判定当前RE串是否被接受，以及给NFA连边。

## 二、求出对应的图

要连的边我们都会连的啦……就是教材的内容，~~相信对图灵机编程技巧比c++还熟悉的你们来说这不成问题~~。然后**用强归纳法（点的个数）可以轻易证明这个图是平面图**，所以我希望能可视化这个图。但是由于可视化库networkx的种种限制，我们不得不自己计算每个点在matplotlib里展示的位置，这在我的乐色代码里就是xlevels（屏幕从左到右）和ylevels（屏幕从上到下）。我之前尝试过在构图后更新，大失败，所以选择在构图过程中算位置。这个愚蠢的决定增加了代码的耦合度，是我代码看上去就很乐色的根源QAQ

两个levels的计算方式是“刷表法”，~~说白了就是暴力更新~~。我们用bfs遍历有向图要更新的那部分（连边前是相对独立的，所以放心遍历），然后计算xlevels和ylevels的增量。

经过无数尝试，我采用了以下定义，可以达到一个不错的可视化观赏效果。”三“会说为什么这样定义：

- 单个字符（base）：xlevel取1和2，ylevel都取1。

- 克林闭包（star）：右侧新点的xlevel = 原右侧点的xlevel + 1，ylevel都取1。

- 连接（concat）：只对右边的那个子图进行xlevel的更新。

- 并集（union）：我们一次合并多个待合并子图（多个空边 = 一个空边，这也是“一”里parse_RE可以写成非递归的原因），所以两侧的ylevel都 = (最下方子图的最大ylevel + 最上方子图的最大ylevel) // 2。右侧新点xlevel = max(各子图的最大xlevel) + 1。


```python
class MyGraph():
    def __init__(self, n):
        self.G = [[] for i in range(n + 5)]
        self.xlevels = [0 for i in range(n + 5)]
        self.ylevels = [0 for i in range(n + 5)]
        self.n = 0

    def get_node_with_max_ylevel(self, st):
        q = queue.Queue()
        q.put(st)
        vis = [False for _ in range(len(self.G))]
        ans = 0
        while not q.empty():
            u = q.get()
            q.task_done()
            if vis[u]:
                continue
            vis[u] = True
            ans = max(ans, self.ylevels[u])
            for nxt in g.G[u]:
                q.put(nxt[0])
        return ans

    def update_levels(self, levels, st, offset):
        q = queue.Queue()
        q.put(st)
        vis = [False for _ in range(len(self.G))]
        while not q.empty():
            u = q.get()
            q.task_done()
            if vis[u]:
                continue
            vis[u] = True
            levels[u] += offset
            for nxt in g.G[u]:
                q.put(nxt[0])

    def addedge(self, p1, p2, w):
        self.G[p1].append((p2, w))

    def addedge_base(self, op):
        self.n += 2
        n = self.n
        self.addedge(n - 1, n, op)
        self.xlevels[n - 1] = 1
        self.xlevels[n] = 2
        self.ylevels[n - 1], self.ylevels[n] = 1, 1
        return n - 1, n

    def addedge_star(self, st, ed):
        self.n += 2
        n = self.n
        self.update_levels(self.xlevels, st, 1)
        self.addedge(ed, st, "ε")
        self.addedge(ed, n, "ε")
        self.addedge(n - 1, st, "ε")
        self.addedge(n - 1, n, "ε")
        self.xlevels[n - 1], self.xlevels[n] = 1, self.xlevels[ed] + 1
        self.ylevels[n - 1], self.ylevels[n] = 1, 1
        return n - 1, n

    def addedge_concat(self, st1, ed1, st2, ed2):
        self.addedge(ed1, st2, "ε")
        self.update_levels(self.xlevels, st2, self.xlevels[ed1])
        return st1, ed2

    def addedge_union(self, sts, eds):
        self.n += 2
        n = self.n
        offsets = [self.get_node_with_max_ylevel(sts[0])]
        for j in range(1, len(sts) - 1):
            offset = self.get_node_with_max_ylevel(sts[j])
            offsets.append(offset + offsets[-1])
        for j in range(1, len(sts)):
            self.update_levels(self.ylevels, sts[j], offsets[j - 1])
        new_ylevel = (self.get_node_with_max_ylevel(sts[0]) + self.get_node_with_max_ylevel(sts[-1])) // 2
        self.ylevels[n - 1], self.ylevels[n] = new_ylevel, new_ylevel
        for ed in eds:
            self.addedge(ed, n, "ε")
        for st in sts:
            self.addedge(n - 1, st, "ε")
        self.update_levels(self.xlevels, n - 1, 1)
        self.xlevels[n - 1], self.xlevels[n] = 1, max([self.xlevels[ed] for ed in eds]) + 1
        return n - 1, n
```



## 三、画图

一开始用matplotlib画的，但是那个注释不能放在箭头的中间QAQ。所以下载了networkx，本来想直接用netx.planar_layout()的，结果networkx没法自己找出平面图的“好平面布局”，所以以上代码我就自己算那个posG了qwq。

**顺便展示下我构造”二“里的递归定义的过程8。**

1. 之前只想到算xlevel，但是这样依然会造成不少的边相交的情况。所以就多算一个ylevel。用了ylevel以后效果好了些。
2. 但是有时候感觉union里面的两侧新点太低了，不好看，所以union情况下，ylevel就改成了下方子图和上方子图最高高度（ylevel）求平均。
3. 可以看到star两侧新点的ylevel定义为1，改其他值可能好看一点。这个留给~~秒解”归约到NP难问题“的~~你们来补上。



之后又发现有向边的环长为2时会重叠。折腾一天才想到：尝试用带弧形的边来解决。但是全都弯的也不行，所以采用了这个分类标准：xlevel相隔1以上的都展示弯边，xlevel相隔为1的就看有没有长为2的环，有则一个弯一个不弯，否则都是直的。

最后还剩一些小问题：标签居然不能跟着弯边走，而是停留在两点的直线上qwqqwq，这样有些标记会互相覆盖。查了一天官方文档、StackOverflow汗百度都没解决办法。并且边的弯曲一点灵活性都没得，明明是平面图却不会“避嫌”的。算了就酱紫吧QAQ。

这部分代码不贴了，看完整代码8。



然后是部分效果展示图：

```python
a*.(b+c).(d+e)+(g*+f*).b*+(a*+b+c)+(b+c*+d)+(d*+e+f)
```

![a幂.(b+c).(d+e)+(g幂+f幂).b幂+(a幂+b+c)+(b+c幂+d)+(d幂+e+f)](./a幂.(b+c).(d+e)+(g幂+f幂).b幂+(a幂+b+c)+(b+c幂+d)+(d幂+e+f).png)

```python
a*.b*.c*+b.c.a+c*.a*.b+d*.(e*+f*)+(g*+h*).i
```

![a幂.b幂.c幂+b.c.a+c幂.a幂.b+d幂.(e幂+f幂)+(g幂+h幂).i](./a幂.b幂.c幂+b.c.a+c幂.a幂.b+d幂.(e幂+f幂)+(g幂+h幂).i.png)

```python
(a+b)*+(c+d+e)*
```

![(a+b)幂+(c+d+e)幂](./(a+b)幂+(c+d+e)幂.png)



接下来是我完整的乐色代码：

```python
import queue
import matplotlib.pyplot as plt
import networkx as netx
import numpy as np
import networkx.drawing.nx_pylab as nxdraw

i = 0
s = ""
token = ""
g = None


class MyGraph():
    def __init__(self, n):
        self.G = [[] for i in range(n + 5)]
        self.xlevels = [0 for i in range(n + 5)]
        self.ylevels = [0 for i in range(n + 5)]
        self.n = 0

    def get_node_with_max_ylevel(self, st):
        q = queue.Queue()
        q.put(st)
        vis = [False for _ in range(len(self.G))]
        ans = 0
        while not q.empty():
            u = q.get()
            q.task_done()
            if vis[u]:
                continue
            vis[u] = True
            ans = max(ans, self.ylevels[u])
            for nxt in g.G[u]:
                q.put(nxt[0])
        return ans

    def update_levels(self, levels, st, offset):
        q = queue.Queue()
        q.put(st)
        vis = [False for _ in range(len(self.G))]
        while not q.empty():
            u = q.get()
            q.task_done()
            if vis[u]:
                continue
            vis[u] = True
            levels[u] += offset
            for nxt in g.G[u]:
                q.put(nxt[0])

    def addedge(self, p1, p2, w):
        self.G[p1].append((p2, w))

    def addedge_base(self, op):
        self.n += 2
        n = self.n
        self.addedge(n - 1, n, op)
        self.xlevels[n - 1] = 1
        self.xlevels[n] = 2
        self.ylevels[n - 1], self.ylevels[n] = 1, 1
        return n - 1, n

    def addedge_star(self, st, ed):
        self.n += 2
        n = self.n
        self.update_levels(self.xlevels, st, 1)
        self.addedge(ed, st, "ε")
        self.addedge(ed, n, "ε")
        self.addedge(n - 1, st, "ε")
        self.addedge(n - 1, n, "ε")
        self.xlevels[n - 1], self.xlevels[n] = 1, self.xlevels[ed] + 1
        self.ylevels[n - 1], self.ylevels[n] = 1, 1
        return n - 1, n

    def addedge_concat(self, st1, ed1, st2, ed2):
        self.addedge(ed1, st2, "ε")
        self.update_levels(self.xlevels, st2, self.xlevels[ed1])
        return st1, ed2

    def addedge_union(self, sts, eds):
        self.n += 2
        n = self.n
        offsets = [self.get_node_with_max_ylevel(sts[0])]
        for j in range(1, len(sts) - 1):
            offset = self.get_node_with_max_ylevel(sts[j])
            offsets.append(offset + offsets[-1])
        for j in range(1, len(sts)):
            self.update_levels(self.ylevels, sts[j], offsets[j - 1])
        new_ylevel = (self.get_node_with_max_ylevel(sts[0]) + self.get_node_with_max_ylevel(sts[-1])) // 2
        self.ylevels[n - 1], self.ylevels[n] = new_ylevel, new_ylevel
        for ed in eds:
            self.addedge(ed, n, "ε")
        for st in sts:
            self.addedge(n - 1, st, "ε")
        self.update_levels(self.xlevels, n - 1, 1)
        self.xlevels[n - 1], self.xlevels[n] = 1, max([self.xlevels[ed] for ed in eds]) + 1
        return n - 1, n


def advance():
    global i, s, token
    if i >= len(s):
        token = None
    else:
        token = s[i]
        i += 1
    return token


def R(st, ed):
    global i, s, token, g
    advance()
    while token == "*":
        st, ed = g.addedge_star(st, ed)
        advance()
    return st, ed


def E():
    global i, s, token, g
    ch = advance()
    if ch and (ch.islower() or ch == "ε" or ch.isdigit()):
        st, ed = g.addedge_base(ch)
    elif ch and ch == "(":
        fl, st, ed = parse_RE()
        if not fl or token != ")":
            return False, 0, 0
    else:
        return False, 0, 0
    st, ed = R(st, ed)
    return True, st, ed


def T():
    global i, s, token, g
    fl1, st1, ed1 = E()
    fl2 = True
    if token == '.':
        fl2, st2, ed2 = T()
        st1, ed1 = g.addedge_concat(st1, ed1, st2, ed2)
    return (fl1 and fl2), st1, ed1


def parse_RE():
    global i, s, token, g
    fl1, st1, ed1 = T()
    st, ed = [st1], [ed1]
    fl2 = True
    while token == "+":
        fl2, st2, ed2 = T()
        fl1 = fl1 and fl2
        if not fl1:
            return fl1, 0, 0
        st.append(st2)
        ed.append(ed2)
    if len(st) > 1:
        st1, ed1 = g.addedge_union(st, ed)
    return fl1, st1, ed1


def allocate_pos():
    pos = {j: np.array([0, 0]) for j in range(1, g.n + 1)}
    for u in range(1, g.n + 1):
        pos[u] = np.array([-1 + 0.2 * g.xlevels[u], -1 + 0.2 * g.ylevels[u]])
    return pos


class Drawer():
    def __init__(self, posG):
        self.G_straight = []
        self.G_curve = []
        self.posG = posG

    def get_conn_style(self):
        for u in range(1, g.n + 1):
            for j in range(len(g.G[u])):
                v = g.G[u][j][0]
                if abs(g.xlevels[u] - g.xlevels[v]) > 1:
                    self.G_curve.append((u, v))
                elif u in [edg[0] for edg in g.G[v]]:
                    if u < v:
                        self.G_straight.append((u, v))
                    else:
                        self.G_curve.append((u, v))
                else:
                    self.G_straight.append((u, v))

    def draw(self):
        G = netx.DiGraph()
        for u in range(1, g.n + 1):
            G.add_node(u, desc=str(u))
        for u in range(1, g.n + 1):
            for j in range(len(g.G[u])):
                v = g.G[u][j][0]
                G.add_edge(u, v, weight=g.G[u][j][1])
        self.get_conn_style()
        nxdraw.draw_networkx_nodes(G, self.posG)
        nxdraw.draw_networkx_edges(G, self.posG, edgelist=self.G_straight)
        nxdraw.draw_networkx_edges(G, self.posG, edgelist=self.G_curve, connectionstyle='arc3,rad = 0.2')
        node_labels = netx.get_node_attributes(G, 'desc')
        netx.draw_networkx_labels(G, pos=self.posG, labels=node_labels)
        edge_labels = netx.get_edge_attributes(G, 'weight')
        netx.draw_networkx_edge_labels(G, pos=self.posG, edge_labels=edge_labels)
        plt.show()


def main():
    global i, s, token, g
    while True:
        i = 0
        s = input()
        token = None
        g = MyGraph(10000)
        fl, st, ed = parse_RE()
        print(fl)  # dbg
        if not fl:
            continue
        print("xlevels", g.xlevels[:g.n + 1])  # dbg
        print("ylevels", g.ylevels[:g.n + 1])  # dbg
        posG = allocate_pos()
        print(posG)  # dbg
        drawer = Drawer(posG)
        drawer.draw()


if __name__ == "__main__":
    main()
```


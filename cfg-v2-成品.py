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

import networkx as nx
import matplotlib.pyplot as plt
import copy


def leaves(net):
    return [u for u in net.nodes() if net.out_degree(u) == 0]

def isCher(tree,lSet,cher):
    node1 = cher[0]
    node2 = cher[1]
    if node1 not in lSet:
        return False
    if node2 not in lSet:
        return False
    par1 = list(tree.predecessors(node1))[0]
    par2 = list(tree.predecessors(node2))[0]
    return par1 == par2

def cherry_in_all_trees(TSet,lSet):
    lvs = lSet[-1].copy()
    setall = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    set = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    for cher in setall: # doet dubbel voor (x,y) en (y,x)
        for i in TSet:
            if not isCher(TSet[i],lSet[i],cher):
                set.remove(cher)
                break
    return set





def pickCher(Tset,lSet,leaf):
    #print(leaf)
    for i in Tset:
        #subax4 = plt.subplot(121)
        #pos = nx.kamada_kawai_layout(Tset[i])
        #nx.draw(Tset[i], pos=pos)
        #nx.draw_networkx_labels(Tset[i], pos, font_size=10, font_family="sans-serif")
        if leaf in lSet[i]:

            p = list(Tset[i].predecessors(leaf))[0]
            if len(list(Tset[i].predecessors(p))) == 0:
                # no grandparent
                Tset[i].remove_node(leaf)
                Tset[i].remove_node(p)
            else:
                gp = list(Tset[i].predecessors(p))[0]
                chldren = list(Tset[i].successors(p))
                chldren.remove(leaf)
                Tset[i].remove_node(leaf)
                Tset[i].remove_node(p)
                Tset[i].add_edge(gp,chldren[0])


        #subax4 = plt.subplot(122)
        #pos = nx.kamada_kawai_layout(Tset[i])
        #nx.draw(Tset[i], pos=pos)
        #nx.draw_networkx_labels(Tset[i], pos, font_size=10, font_family="sans-serif")
        #plt.show()

def canPick(TreeSet,lSet):
    lvs = lSet[-1].copy()
    DelSet = []
    for leaf in lvs:
        for i in TreeSet:
            if leaf in lSet[i]:
                parent = list(TreeSet[i].predecessors(leaf))[0]
                chldren = list(TreeSet[i].successors(parent))
                chldren.remove(leaf)
                if chldren[0] not in lvs:
                    DelSet.append(leaf)
                    break
    for leaf in DelSet:
        lvs.remove(leaf)
    return lvs

def inCher(TreeSet,lSet,leaf):
    lvs = lSet[-1].copy()
    lvs.remove(leaf)
    cherSet = []

    for i in TreeSet:
        if leaf in lSet[i]:
            parent = list(TreeSet[i].predecessors(leaf))[0]
            chldren = list(TreeSet[i].successors(parent))
            chldren.remove(leaf)
            if chldren[0] in lvs:
                if chldren[0] not in cherSet:
                    cherSet.append(chldren[0])
    return cherSet




def lecSolver(T):
    # T: set of trees
    # s: sequence
    # k
    sol = []
    s = []
    check = []
    w = 0

    lSet = dict()
    totLeaves = []
    for i in T:
        lSet[i] = leaves(T[i])
        for j in lSet[i]:
            if j not in totLeaves:
                totLeaves.append(j)
    lSet[-1] = totLeaves
    k = len(lSet[-1])**2
    check.append([T,s,w,lSet])
    #print(lSet)


    while len(check) > 0:
        T1 = {}
        for i in check[0][0]:
            if len(check[0][3][i]) > 1:
                T1[i] = copy.deepcopy(check[0][0][i])
        s1 = check[0][1].copy()
        w1 = check[0][2]
        lSet1 = {}
        for i in check[0][3]:
            if len(check[0][3][i]) > 1 or i < 0 :
                lSet1[i] = copy.deepcopy(check[0][3][i])
        check.remove(check[0])

        #print(lSet1, "check")
        #print("check ", s1 ,len(T1))

        if w1 > k:
            #print("too many constraints ", w, k)
            continue

        if len(lSet1[-1]) == 1:
            s1.append(lSet1[-1][0])
            #print("found solution ", w1, s1, len(check))
            if w1 < k:
                k = w1
                sol = []

            sol.append([w1,s1])
            continue

        cherries = cherry_in_all_trees(T1,lSet1)
        if len(cherries) > 0:
            pickCher(T1,lSet1,cherries[0][0])
            s1.append(cherries[0][0])
            #print("remove leaf ", cherries[0][0], " sol ", s1)
            for i in lSet1:
                lSet1[i].remove(cherries[0][0])
            #print("cherrinall")
            check.insert(0,[T1, s1, w1, lSet1])
            continue

        pickLeaves = canPick(T1,lSet1)
        #print(pickLeaves)
        #if len(pickLeaves) == 0:
            #print("No picks")
        #print(len(pickLeaves), pickLeaves)
        for leaf in pickLeaves:
            newS = s1.copy()
            newW = w1 + len(inCher(T1,lSet, leaf))-1
            #print(inCher(T1, leaf))
            newT = {}
            for i in T1:
                newT[i] = copy.deepcopy(T1[i])
            newLSet = {}
            for i in lSet1:
                sub = lSet1[i].copy()
                if leaf in sub:
                    sub.remove(leaf)
                newLSet[i] = sub
            pickCher(newT,lSet1,leaf)
            newS.append(leaf)
            check.insert(0, [newT, newS, newW, newLSet])
            #print("next" , newS, newW)



    return sol













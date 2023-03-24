import networkx as nx
import copy


def leaves(net):
    return [u for u in net.nodes() if net.out_degree(u) == 0]

def isCher(tree,cher):
    par1 = list(tree.predecessors(cher[0]))[0]
    par2 = list(tree.predecessors(cher[1]))[0]
    return par1 == par2

def cherry_in_all_trees(TSet):
    lvs = leaves(TSet[0])
    setall = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    set = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    for cher in setall: # doet dubbel voor (x,y) en (y,x)
        for i in TSet:
            if not isCher(TSet[i],cher):
                set.remove(cher)
                break
    return set





def pickCher(Tset,leaf):
    #print(leaf)
    for i in Tset:
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



def canPick(TreeSet):
    lvs = leaves(TreeSet[0])
    DelSet = []
    for leaf in lvs:
        for i in TreeSet:
            parent = list(TreeSet[i].predecessors(leaf))[0]
            chldren = list(TreeSet[i].successors(parent))
            chldren.remove(leaf)
            if chldren[0] not in lvs:
                DelSet.append(leaf)
                break
    for leaf in DelSet:
        lvs.remove(leaf)
    return lvs



def checkTemporal(T):
    # T: set of trees
    # s: sequence
    # k

    s = []
    check = []

    check.append([T,s])


    while len(check) > 0:
        T1 = {}
        for i in check[0][0]:
            T1[i] = copy.deepcopy(check[0][0][i])
        s1 = check[0][1].copy()
        check.remove(check[0])

        #print(lSet1, "check")

        if len(leaves(T1[0])) == 1:
            s1.append(leaves(T1[0])[0])
            #print("found solution ", s1, len(check))
            return True

        cherries = cherry_in_all_trees(T1)
        if len(cherries) > 0:
            pickCher(T1,cherries[0][0])
            s1.append(cherries[0][0])
            #print("remove leaf ", cherries[0][0], " sol ", s1)
            #print("cherrinall")
            check.insert(0,[T1, s1])
            continue

        pickLeaves = canPick(T1)
        #print(pickLeaves)
        #if len(pickLeaves) == 0:
            #print("No picks")
        #print(len(pickLeaves), pickLeaves)
        for leaf in pickLeaves:
            newS = s1.copy()
            newT = {}
            for i in T1:
                newT[i] = copy.deepcopy(T1[i])
            pickCher(newT,leaf)
            newS.append(leaf)
            check.insert(0, [newT, newS])
            #print("next" , newS, newW)
    return False













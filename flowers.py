import numpy as np
from collections import Counter


def get_color(fls):
    """
    Get the color given the genes

    0 - weiß
    1 - schwarz
    2 - pink
    3 - violet
    4 - rot
    5 - gelb
    6 - orange
    7 - blau
    8 - grün
    """
    if not isinstance(fls, list):
        fls = [fls]
    if isinstance(fls[0], Flower):
        fls = list(map(lambda f: f.genes, fls))

    if len(fls[0]) == 4:
        i, j, k, l = np.transpose(fls)
        return colormap[typ][i, j, k, l]
    if len(fls[0]) == 3:
        i, j, k = np.transpose(fls)
        return colormap[typ][i, j, k]
    else:
        raise ValueError


class Flower:

    def __init__(self, genes, step=0, power=0, parent_genes=None):
        self.genes = genes
        self.step = step
        self.power = power
        self.parent_genes = parent_genes
        self.color_dict = {0: 'white', 1: 'black', 2: 'pink', 3: 'purple', 4: 'red',
                           5: 'yellow', 6: 'orange', 7: 'blue', 8: 'green'}

    def __repr__(self):
        return "<%s | %s | %d %d>" % (self.genes, self.color_dict[int(get_color(self.genes))], self.step, self.power)

    def prnt(self):
        return "<%s | %6s | %d %2d <- %s>" % (self.genes, self.color_dict[int(get_color(self.genes))],
                                              self.step, self.power, self.parent_genes or 'Seed')


class Breeder:

    def __init__(self, typ='rose'):
        self.pool = {fl.genes: fl for fl in seeds[typ]}

    def result(self, small=True):
        seen = set()
        fls = set()
        for fl in sorted(self.pool.values(), key=lambda fl: fl.power):
            fl_c = int(get_color(fl))
            if fl_c not in seen:
                seen.add(fl_c)
                fls.add(fl)
                print(fl.prnt())
            elif small is False:
                print(fl.prnt())

        print("-----")
        gns = set([fl.genes for fl in fls])
        # Also the needed ones
        if small:
            todo = set([p for fl in fls if fl.parent_genes is not None for p in fl.parent_genes if p not in gns])
            while len(todo) > 0:
                p = todo.pop()
                pa = self.pool[p]
                print(pa.prnt())
                for par in pa.parent_genes:
                    if par not in gns:
                        todo.add(par)

    def all_pos(self):
        old_pool_len = 0
        while len(self.pool) != old_pool_len:
            old_pool_len = len(self.pool)
            self.step()

    def step(self):
        pool_fls = list(self.pool.values())
        new_pool = []
        for i in range(len(pool_fls)):
            for j in range(i, len(pool_fls)):
                new_pool += self.breed(pool_fls[i], pool_fls[j])

        for fl in new_pool:
            if fl.genes not in self.pool:
                self.pool[fl.genes] = fl
            else:
                if fl.power < self.pool[fl.genes].power:
                    self.pool[fl.genes] = fl

    def breed(self, fl1, fl2):
        childs, probs = self.crossover(fl1, fl2)
        # print(fl1, fl2, childs, probs)
        cols = get_color(childs)
        cnts = Counter(cols)
        unique_cols = [c for c in cnts if cnts[c] == 1]
        return [Flower(tuple(ch), step=max(fl1.step, fl2.step) + 1, power=fl1.power + fl2.power + 1/pr,
                       parent_genes=(fl1.genes, fl2.genes))
                for ch, pr in zip(childs, probs) if get_color(tuple(ch)) in unique_cols]

    def crossover(self, fl1, fl2):
        """
        returns offspring genes and probabilities
        """
        out = [[]]
        ch = [1.0]
        for g1, g2 in zip(fl1.genes, fl2.genes):
            nout = []
            nch = []
            for o, p in zip(out, ch):
                if g1 == 1 and g2 == 1:
                    nout += [o + [0], o + [1], o + [2]]
                    nch += [0.25 * p, 0.5 * p, 0.25 * p]
                elif g1 == 1 or g2 == 1:
                    nout += [o + [g1], o + [g2]]
                    nch += [0.5 * p, 0.5 * p]
                else:
                    nout += [o + [int(min(g1, g2) + abs(g2 - g1)/2)]]
                    nch += [p]
            out = nout
            ch = nch
        return out, ch


colormap = {'rose': np.array(
                [[[[3, 3, 3], [0, 0, 0], [0, 0, 0]],
                  [[3, 3, 3], [0, 0, 0], [5, 5, 5]],
                  [[0, 0, 0], [5, 5, 5], [5, 5, 5]]],
                 [[[4, 2, 0], [4, 2, 0], [4, 2, 0]],
                  [[4, 2, 3], [4, 2, 0], [6, 5, 5]],
                  [[4, 2, 0], [6, 5, 5], [6, 5, 5]]],
                 [[[1, 4, 2], [1, 4, 2], [1, 4, 2]],
                  [[1, 4, 3], [4, 4, 0], [6, 6, 5]],
                  [[7, 4, 0], [6, 6, 5], [6, 6, 5]]]]),
            'cosmos': np.array(
                [[[0, 0, 0], [5, 5, 0], [5, 5, 5]],
                 [[2, 2, 2], [6, 6, 2], [6, 6, 6]],
                 [[4, 4, 4], [6, 6, 4], [1, 1, 4]]]),
            'lilly': np.array(
                [[[0, 0, 0], [5, 0, 0], [5, 5, 0]],
                 [[4, 2, 0], [6, 5, 5], [6, 5, 5]],
                 [[1, 4, 2], [1, 4, 2], [6, 6, 0]]]),
            'pansy': np.array(
                [[[7, 0, 0], [7, 5, 5], [5, 5, 5]],
                 [[7, 4, 4], [6, 6, 6], [5, 5, 5]],
                 [[3, 4, 4], [3, 4, 4], [3, 6, 6]]]),
            'hyacinth': np.array(
                [[[7, 0, 0], [0, 5, 5], [5, 5, 5]],
                 [[0, 2, 4], [5, 5, 6], [5, 5, 6]],
                 [[4, 4, 4], [4, 4, 7], [3, 3, 3]]]),
            'tulip': np.array(
                [[[0, 0, 0], [5, 5, 0], [5, 5, 5]],
                 [[4, 2, 0], [6, 5, 5], [6, 5, 5]],
                 [[1, 4, 4], [1, 4, 4], [3, 3, 3]]]),
            'mum': np.array(
                [[[3, 0, 0], [0, 5, 5], [5, 5, 5]],
                 [[2, 2, 2], [2, 4, 5], [3, 3, 3]],
                 [[4, 4, 4], [4, 3, 3], [4, 8, 8]]]),
            'windflower': np.array(
                [[[7, 0, 0], [7, 6, 6], [6, 6, 6]],
                 [[7, 4, 4], [2, 2, 2], [6, 6, 6]],
                 [[3, 4, 4], [3, 4, 4], [3, 2, 2]]]
            )}

seeds = {'rose': [Flower((0, 0, 1, 0)), Flower((0, 2, 2, 0)), Flower((2, 0, 2, 1))],
         'cosmos': [Flower((0, 0, 1)), Flower((0, 2, 1)), Flower((2, 0, 0))],
         'lilly': [Flower((0, 0, 2)), Flower((0, 2, 0)), Flower((2, 0, 1))],
         'pansy': [Flower((0, 0, 1)), Flower((0, 2, 2)), Flower((2, 0, 2))],
         'hyacinth': [Flower((0, 0, 1)), Flower((0, 2, 2)), Flower((2, 0, 1))],
         'tulip': [Flower((0, 0, 1)), Flower((0, 2, 0)), Flower((2, 0, 1))],
         'mum': [Flower((0, 0, 1)), Flower((0, 2, 2)), Flower((2, 0, 2))],
         'windflower': [Flower((0, 0, 1)), Flower((0, 2, 2)), Flower((2, 0, 2))]}

if __name__ == '__main__':
    for typ in seeds.keys():
        print(typ)
        br = Breeder(typ=typ)
        br.all_pos()
        br.result()
        print('*****')

    print('~~~~~')

    for typ in seeds.keys():
        print(typ)
        br = Breeder(typ=typ)
        br.all_pos()
        br.result(small=False)
        print('*****')

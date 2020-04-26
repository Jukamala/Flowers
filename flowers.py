import numpy as np
from collections import Counter


def get_color(fls, typ):
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

    def __init__(self, typ, genes, step=0, power=0, tests=0, parent_genes=None, testing=None):
        self.typ = typ
        self.genes = genes
        self.step = step
        self.power = power
        self.tests = tests
        self.parent_genes = parent_genes
        self.color_dict = {0: 'white', 1: 'black', 2: 'pink', 3: 'purple', 4: 'red',
                           5: 'yellow', 6: 'orange', 7: 'blue', 8: 'green'}
        self.test_seed, self.ac_cols, self.rej_cols, self.alt = testing or 4 * [None]

    def __repr__(self):
        return "<%s | %s | %d %d (%d)>" % (self.genes, self.color_dict[int(get_color(self.genes, self.typ))],
                                           self.step, self.power, self.tests)

    def prnt(self):
        return "<%s | %6s | %d %2d <- %s%s>" %\
               (self.genes, self.color_dict[int(get_color(self.genes, self.typ))], self.step, self.power,
                self.parent_genes or 'Seed',
                "" if self.test_seed is None else "[alt:%s breed with %s for %s]" %
                                                  (self.alt, self.test_seed, self.ac_cols))


class Breeder:

    def __init__(self, typ='rose', test=0):
        """
        test - 0: no tests
             - 1: tests that work after 1st try with seed flower
             - 2: tests that work after 1st try with flowers in pool
             - 3: tests that work on correct flower after 1st try with seed flower
             - 4: tests that work on correct flower after 1st try with flowers in pool
        """
        self.typ = typ
        self.seeds = seeds[typ]
        self.pool = {fl.genes: fl for fl in seeds[typ]}
        self.test = test

    def result(self, small=True):
        seen = set()
        fls = set()
        for fl in sorted(self.pool.values(), key=lambda fl: fl.power):
            fl_c = int(get_color(fl, self.typ))
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
        if fl1.typ != fl2.typ:
            raise ValueError

        childs, probs = self.crossover(fl1, fl2)
        # print(fl1, fl2, childs, probs)
        cols = get_color(childs, typ=self.typ)
        cnts = Counter(cols)
        unique_cols = [c for c in cnts if cnts[c] == 1]
        fls = [Flower(fl1.typ, tuple(ch), step=max(fl1.step, fl2.step) + 1, power=fl1.power + fl2.power + 1/pr,
                      parent_genes=(fl1.genes, fl2.genes), tests=fl1.tests + fl2.tests)
               for ch, pr in zip(childs, probs) if get_color(tuple(ch), self.typ) in unique_cols]

        if self.test >= 1:
            # 1st try seed test
            testable_cols = [c for c in cnts if cnts[c] == 2]
            test_sets = {c: [] for c in testable_cols}
            for ch, pr in zip(childs, probs):
                cc = int(get_color(tuple(ch), self.typ))
                if cc in testable_cols:
                    test_sets[cc] += [(ch, pr)]

            for (ch1, pr1), (ch2, pr2) in test_sets.values():
                goodness = 10
                testseed = None
                ac_cols = None
                rej_cols = None
                alt = None
                for seed in self.seeds if self.test == 1 else self.pool.values():
                    cols1, _ = self.crossover(tuple(ch1), seed)
                    cols2, _ = self.crossover(tuple(ch2), seed)
                    cols1 = list(map(tuple, cols1))
                    cols2 = list(map(tuple, cols2))
                    if self.test <= 2:
                        # 1st try
                        if len(set(cols1) & set(cols2)) == 0:
                            # Good test
                            gd = len(set(cols1) | set(cols2))
                            if gd < goodness:
                                testseed = seed
                                ac_cols = set(cols1)
                                rej_cols = set(cols2)
                        # 1st try on correct
                        else:
                            pass
                if testseed is not None:
                    print(fl1, fl2, testseed, get_color(tuple(ch1), self.typ))
                    fls += [Flower(fl1.typ, tuple(ch1), step=max(fl1.step, fl2.step, testseed.step) + 1,
                                   parent_genes=(fl1.genes, fl2.genes),
                                   power=fl1.power + fl2.power + testseed.power + 1/pr1 + ,
                                   tests=fl1.tests + fl2.tests + testseed.tests + 1,
                                   testing=[testseed.genes, ac_cols, rej_cols, ch2])]
                    fls += [Flower(fl1.typ, tuple(ch2), step=max(fl1.step, fl2.step, testseed.step) + 1,
                                   parent_genes=(fl1.genes, fl2.genes),
                                   power=fl1.power + fl2.power + testseed.power + ((pr1 + pr2) / pr1**2),
                                   tests=fl1.tests + fl2.tests + testseed.tests + 1,
                                   testing=[testseed.genes, rej_cols, ac_cols, ch1])]

        return fls

    def crossover(self, fl1, fl2):
        """
        returns offspring genes and probabilities
        """
        if isinstance(fl1, Flower):
            fl1 = fl1.genes
        if isinstance(fl2, Flower):
            fl2 = fl2.genes

        out = [[]]
        ch = [1.0]
        for g1, g2 in zip(fl1, fl2):
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

seeds = {'rose': [Flower('rose', (0, 0, 1, 0)), Flower('rose', (0, 2, 2, 0)), Flower('rose', (2, 0, 2, 1))],
         'cosmos': [Flower('cosmos', (0, 0, 1)), Flower('cosmos', (0, 2, 1)), Flower('cosmos', (2, 0, 0))],
         'lilly': [Flower('lilly', (0, 0, 2)), Flower('lilly', (0, 2, 0)), Flower('lilly', (2, 0, 1))],
         'pansy': [Flower('pansy', (0, 0, 1)), Flower('pansy', (0, 2, 2)), Flower('pansy', (2, 0, 2))],
         'hyacinth': [Flower('hyacinth', (0, 0, 1)), Flower('hyacinth', (0, 2, 2)), Flower('hyacinth', (2, 0, 1))],
         'tulip': [Flower('tulip', (0, 0, 1)), Flower('tulip', (0, 2, 0)), Flower('tulip', (2, 0, 1))],
         'mum': [Flower('mum', (0, 0, 1)), Flower('mum', (0, 2, 2)), Flower('mum', (2, 0, 2))],
         'windflower': [Flower('windflower', (0, 0, 1)), Flower('windflower', (0, 2, 2)),
                        Flower('windflower', (2, 0, 2))]}

if __name__ == '__main__':

    br = Breeder('hyacinth', test=3)
    f1 = Flower('hyacinth', (0, 0, 1))
    brrr = br.breed(f1, f1)
    [print(f.prnt()) for f in brrr]
    for typ in seeds.keys():
        print(typ)
        br = Breeder(typ=typ, test=2)
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


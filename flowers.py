import numpy as np
from collections import Counter


def average_breed_tries_comp(fl1, fl2):
    """
    Create an order on flowers which prefers flowers that
    need a smaller number of breeding tries to breed (in average).

    This is the order used by standard.

    Outputs True if fl1 is better than fl2
    """
    order = ['power', 'tests', 'step']
    for crit in order:
        if getattr(fl1, crit) < getattr(fl2, crit):
            return True
        elif getattr(fl1, crit) > getattr(fl2, crit):
            return False
    return False


def get_color(fls, typ):
    """
    Input:
    fls - List of Flowers or flower genes
    typ - the type of the flower

    Output:
    A list containing numbers representing the colors:
    0 - white
    1 - black
    2 - pink
    3 - purple
    4 - red
    5 - yellow
    6 - orange
    7 - blue
    8 - green
    """
    # Convert to list of genes
    if not isinstance(fls, list):
        fls = [fls]
    if isinstance(fls[0], Flower):
        fls = list(map(lambda f: f.genes, fls))

    # Search in the colormap given below
    if len(fls[0]) == 4:
        i, j, k, l = np.transpose(fls)
        return colormap[typ][i, j, k, l]
    if len(fls[0]) == 3:
        i, j, k = np.transpose(fls)
        return colormap[typ][i, j, k]
    else:
        raise ValueError("Wrong flower type %s chose from:\n%s" % (typ, colormap.keys()))


class Flower:

    def __init__(self, typ, genes, step=0, power=0, tests=0, parent_genes=None, testing=None, p=None, generic=None):
        self.typ = typ
        self.genes = genes
        self.step = step
        self.power = power
        self.tests = tests
        self.parent_genes = parent_genes
        self.p = p
        self.generic = generic
        self.color_dict = {0: 'white', 1: 'black', 2: 'pink', 3: 'purple', 4: 'red',
                           5: 'yellow', 6: 'orange', 7: 'blue', 8: 'green'}
        self.test_seed, self.ac_cols, self.rej_cols, self.alt = testing or 4 * [None]

    def __repr__(self):
        genes = self.genes or "Generic"
        col = self.color_dict[self.generic if self.generic is not None else int(get_color(self.genes, self.typ))]
        return "<%s | %s | %d %d (%d)>" % (genes, col, self.step, self.power, self.tests)

    def prnt(self):
        genes = self.genes or "Generic"
        gene_str = "%10s" % str(genes) if self.typ != 'rose' else "%13s" % str(genes)
        col = self.color_dict[self.generic if self.generic is not None else int(get_color(self.genes, self.typ))]
        test_str = "" if self.test_seed is None else " [alt:%s breed with %s for %s%s]" %\
                                                     (self.alt, self.test_seed, self.ac_cols,
                                                      "" if len(self.rej_cols) == 0 else " not %s" % self.rej_cols)
        chance_str = "" if self.p is None else " [%s %%]" % (int(100 * self.p) if (100 * self.p) % 1 < 0.01 else
                                                             "%.1f" % (100 * self.p))
        return "<%s | %6s | %d %2d <- %s%s%s>" %\
               (gene_str, col, self.step, self.power, self.parent_genes or 'Seed', test_str, chance_str)


class Breeder:
    """
    A breeder that can be used to calculate optimal routes.
    It starts with a pool of flowers and combines them iteratively in all possible combinations.

    e.g.:
    '''
    # Start from seeds
    br = Breeder(typ='rose')
    br.all_pos()
    br.result()

    # Start with flowers from islands included and allow tests
    typ = 'rose'
    pool = seeds[typ] + [Flower(typ, (2, 2, 1, 1)), Flower(typ, (2, 0, 0, 2))
    br = Breeder(typ=typ, pool=pool, test=3)
    br.all_pos()
    br.result()

    # Show routes for all hybrids
    br = Breeder(typ='hyacinth', test=3)
    br.all_pos()
    br.result(all=True)
    '''
    """

    def __init__(self, typ='rose', pool=None, test=0, comp=average_breed_tries_comp):
        """
        Parameters:
        typ - The type of the flower, one of:
              [rose, cosmos, lilly, pansy, hyacinth, tulip, mum, windflower]
        pool - The flowers to start with. Defaults to the flowers obtained from seeds
        test - Sometimes a breeding step can get you two genetically different variants.
               A little extra work can be done to determine which one to use. This sets what test are allowed
            0 - no tests
            1 - tests that work after the first try and use seed flower
            2 - tests that work after the first try and use any flowers in pool
            3 - any test that uses seed flowers
            4 - any test that uses any flower in pool
        comp - A function that takes two flowers as input and returns True if the first is better.
               Defaults to a comparison on the average breeding tries.
        """
        self.typ = typ
        self.seeds = seeds[typ]
        self.pool = {fl.genes: fl for fl in pool or seeds[typ]}
        self.test = test
        self.comp = comp

    def result(self, big=False):
        """
        Print the best path to every color that was found
        When big is True paths to all possible hybrids are printed
        """
        seen = set()
        fls = set()
        for fl in sorted(self.pool.values(), key=lambda fl: fl.power):
            fl_c = int(fl.generic if fl.generic is not None else get_color(fl, self.typ))
            if fl_c not in seen:
                seen.add(fl_c)
                fls.add(fl)
                print(fl.prnt())
            elif big:
                print(fl.prnt())

        gns = set([fl.genes for fl in fls])
        todo = set([p for fl in fls if fl.parent_genes is not None for p in fl.parent_genes if p not in gns])
        todo = todo | set([fl.test_seed for fl in fls if fl.test_seed is not None if fl.test_seed not in gns])

        # Also the needed ones
        if not big:
            if len(todo):
                print("-----")
            while len(todo) > 0:
                p = todo.pop()
                pa = self.pool[p]
                print(pa.prnt())
                for par in pa.parent_genes:
                    if par not in gns:
                        todo.add(par)
                if pa.test_seed is not None and pa.test_seed not in gns:
                    todo.add(pa.test_seed)

    def all_pos(self, intermediate_results=False):
        """
        Breed until to breeding can be done to improve the pool of flowers
        """
        while self.step():
            if intermediate_results:
                self.result(big=True)
                print("#")

    def step(self):
        """
        Performs a breeding step on the pool of flowers currently available.
        Breed all pairs of available flowers with known genes.
        Take new flowers in to the pool and update flowers when better origins are found.
        Output:
        Returns if something was changed in this step.
        """
        change = False
        pool_fls = list(self.pool.values())
        new_pool = []
        for i in range(len(pool_fls)):
            for j in range(i, len(pool_fls)):
                if pool_fls[i].genes is not None and pool_fls[j].genes:
                    new_pool += self.breed(pool_fls[i], pool_fls[j])
        for fl in new_pool:
            if (fl.genes or fl.generic) not in self.pool:
                self.pool[fl.genes or fl.generic] = fl
                change = True
            else:
                if self.comp(fl, self.pool[fl.genes or fl.generic]):
                    self.pool[fl.genes or fl.generic] = fl
                    change = True
        return change

    def breed(self, fl1, fl2):
        """
        Input:
        fl1, fl2 - Two flowers to breed
        Dependencies:
        Depends on what test are allowed. See __init__ for more info.
        Output:
        A list of all offspring whose genes are known after breeding.
        (This means this list is incomplete in a sense; e.g.:
         Red and White Roses can create pink roses,
         but there is no easy way to know which genes they have, so they wont be listed here)
        A list of Flower objects are returned - to keep track of parents, chances, testing, etc.
        """
        if fl1.typ != fl2.typ:
            raise ValueError("Can only breed Flowers of the same type")

        # All offspring
        childs, probs = self.crossover(fl1, fl2)
        cols = get_color(childs, typ=self.typ)
        cnts = Counter(cols)
        # Offspring without different variants
        unique_cols = [c for c in cnts if cnts[c] == 1]
        fls = [Flower(fl1.typ, tuple(ch), step=max(fl1.step, fl2.step) + 1,
                      power=fl1.power + (fl2.power if fl1.genes != fl2.genes else 0) + 1/pr,
                      parent_genes=(fl1.genes, fl2.genes), tests=fl1.tests + fl2.tests, p=pr)
               for ch, pr in zip(childs, probs) if get_color(tuple(ch), self.typ) in unique_cols]
        many_cols = [c for c in cnts if cnts[c] >= 1]

        # Generic colors
        col_chances = dict()
        for cc, p_ in zip(cols, probs):
            col_chances[cc] = col_chances.get(cc, 0) + p_

        fls += [Flower(fl1.typ, genes=None, step=max(fl1.step, fl2.step) + 1,
                       power=fl1.power + (fl2.power if fl1.genes != fl2.genes else 0) + 1 / pr,
                       parent_genes=(fl1.genes, fl2.genes), tests=fl1.tests + fl2.tests, p=pr, generic=cc)
                for cc, pr in col_chances.items() if cc in many_cols]

        # If testing is allowed
        if self.test >= 1:
            # Offspring that comes in two variants
            testable_cols = [c for c in cnts if cnts[c] == 2]
            test_sets = {c: [] for c in testable_cols}
            for ch, pr in zip(childs, probs):
                cc = int(get_color(tuple(ch), self.typ))
                if cc in testable_cols:
                    test_sets[cc] += [(ch, pr)]

            # Add other direction for more readable code
            test_sets = {k: [v, [v[1], v[0]]] for k, v in test_sets.items()}

            for (ch1, pr1), (ch2, pr2) in [t for test in test_sets.values() for t in test]:
                # See if it is possible to isolate ch1 from ch2
                # Level of a test is the average tries it takes to determine if a flower is the correct one
                level = float('inf')
                goodness = float('inf')
                test_seed = None
                ac_cols = None
                rej_cols = None
                for seed in self.seeds if self.test in [1, 3] else self.pool.values():
                    # Offspring with the test flower
                    genes1, chances1 = self.crossover(tuple(ch1), seed)
                    genes2, chances2 = self.crossover(tuple(ch2), seed)
                    genes1 = list(map(tuple, genes1))
                    genes2 = list(map(tuple, genes2))
                    cols1 = get_color(genes1, typ=self.typ)
                    cols2 = get_color(genes2, typ=self.typ)
                    if self.test <= 2:
                        # 1st try tests that don't have common colors.
                        if len(set(cols1) & set(cols2)) == 0:
                            # Good test
                            gd = len(set(cols1) | set(cols2))
                            if gd < goodness:
                                level = 1
                                test_seed = seed
                                ac_cols = set(cols1)
                                rej_cols = set(cols2)
                    else:
                        # Other test.
                        # Has accepted colors. When you see them the flower is the correct one.
                        # Has rejected colors. When you see them the flower is the incorrect one.

                        # Chances to get a certain color
                        col_chances1 = dict()
                        col_chances2 = dict()
                        for c, p in zip(genes1, chances1):
                            cc = int(get_color(c, typ=self.typ))
                            col_chances1[cc] = col_chances1.get(cc, 0) + p
                        for c, p in zip(genes2, chances2):
                            cc = int(get_color(c, typ=self.typ))
                            col_chances2[cc] = col_chances2.get(cc, 0) + p

                        # Chance that a random variant breeds with the test flower and creates a flower that
                        # could have come from both variants
                        ths_union = sum([pr1 * col_chances1[x] + pr2 * col_chances2[x]
                                         for x in set(cols1) & set(cols2)]) / (pr1 + pr2)
                        # Chance that a random variant breed with the test flower and from the result
                        # it can be concluded that the variant was indeed <ch>
                        ths_acc = sum([pr1 * col_chances1[x] for x in set(cols1) - set(cols2)]) / (pr1 + pr2)
                        if ths_acc == 0:
                            # Testing is useless
                            continue
                        # The average needed steps to get to a result with this test.
                        ths_level = ths_acc / (1 - ths_union)**2
                        if ths_level < level:
                            level = ths_level
                            test_seed = seed
                            ac_cols = set(cols1) - set(cols2)
                            rej_cols = set(cols2) - set(cols1)

                if test_seed is not None:
                    # Combine power, steps, etc.
                    # Note that breeding with itself doesn't result in extra cost
                    fls += [Flower(fl1.typ, tuple(ch1), step=max(fl1.step, fl2.step, test_seed.step) + 1,
                                   parent_genes=(fl1.genes, fl2.genes),
                                   power=(fl1.power + (fl2.power if fl1.genes != fl2.genes else 0) +
                                          test_seed.power + 1/pr1 + (1 + pr2/pr1) * level),
                                   tests=fl1.tests + fl2.tests + test_seed.tests + 1, p=pr1,
                                   testing=[test_seed.genes, ac_cols, rej_cols, tuple(ch2)])]
                else:
                    fls += [Flower(fl1.typ, genes=None, step=max(fl1.step, fl2.step) + 1,
                                   power=fl1.power + (fl2.power if fl1.genes != fl2.genes else 0) + 1 / (pr1 + pr2),
                                   parent_genes=(fl1.genes, fl2.genes), tests=fl1.tests + fl2.tests, p=pr1 + pr2,
                                   generic=int(get_color(tuple(ch1), typ=self.typ)))]

        return fls

    def crossover(self, fl1, fl2):
        """
        Input:
        fl1, fl2: Two Flowers (or genes of Flowers) to do a crossover
        Output:
        All possible offspring genes with their probabilities
        """
        # Convet to genes if Flowers are given
        if isinstance(fl1, Flower):
            fl1 = fl1.genes
        if isinstance(fl2, Flower):
            fl2 = fl2.genes

        out = [[]]
        ch = [1.0]
        for g1, g2 in zip(fl1, fl2):
            nout = []
            nch = []
            # For every gene
            for o, p in zip(out, ch):
                # Mendelian rules
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

island_seeds = {'rose': [Flower('rose', (0, 0, 1, 0)), Flower('rose', (0, 2, 2, 0)), Flower('rose', (2, 0, 2, 1))],
         'cosmos': [Flower('cosmos', (0, 0, 1)), Flower('cosmos', (0, 2, 1)), Flower('cosmos', (2, 0, 0))],
         'lilly': [Flower('lilly', (0, 0, 2)), Flower('lilly', (0, 2, 0)), Flower('lilly', (2, 0, 1))],
         'pansy': [Flower('pansy', (0, 0, 1)), Flower('pansy', (0, 2, 2)), Flower('pansy', (2, 0, 2))],
         'hyacinth': [Flower('hyacinth', (0, 0, 1)), Flower('hyacinth', (0, 2, 2)), Flower('hyacinth', (2, 0, 1))],
         'tulip': [Flower('tulip', (0, 0, 1)), Flower('tulip', (0, 2, 0)), Flower('tulip', (2, 0, 1))],
         'mum': [Flower('mum', (0, 0, 1)), Flower('mum', (0, 2, 2)), Flower('mum', (2, 0, 2))],
         'windflower': [Flower('windflower', (0, 0, 1)), Flower('windflower', (0, 2, 2)),
                        Flower('windflower', (2, 0, 2))]}


if __name__ == '__main__':

    # Optimal routes with testing
    for typ in seeds.keys():
        print(typ)
        br = Breeder(typ=typ, test=3)
        br.all_pos()
        br.result()
        print('*****')

    print('~~~~~')

    # Optimal routes without testing
    for typ in seeds.keys():
        print(typ)
        br = Breeder(typ=typ, test=0)
        br.all_pos()
        br.result()
        print('*****')

    print('~~~~~')

    for typ in seeds.keys():
        print(typ)
        br = Breeder(typ=typ)
        br.all_pos()
        br.result(big=True)
        print('*****')


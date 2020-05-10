# Optimal flower breeding routes for Animal Crossing: New Horizons.

Breeding flowers in Animal Crossing can be quite complicated.
Especially some hybrids are hard to get by (the best example herefore is the blue roses which is notoriously difficult to bred).

Some currently used routes (see for example the excellent guide by Paleh https://docs.google.com/document/u/0/d/1ARIQCUc5YVEd01D7jtJT9EEJF45m07NXhAm4fOpNvCs/mobilebasic) make use of 'uncertain' breeding steps where you have to test offspring.

(e.g. the famous 'hybrid red' rose can be obtained from orange and purple with 25 % while an 'uncool' red can be obtained with 25 % also. There is no easy way to check what you got)

#### The goal of this project is to give optimal routes to get all flower colors or hybrids.
The route creation is highly customizable:
We start with a pool of flowers. This defaults to the seed flowers. But you can also include mystery island flowers or Handpick genes if you know what to do.
You can tweak what kind of testing is allowed.
No testing (only 'certain' steps), tests where you find out what variant to use with the first
offspring or any testing imaginable - or something between.

I already mentioned that the result is optimal:

### Optimality

When choosing optimal routes you first have to choose what it means for a route to be better than another.
Main characteristics include:

+ **power** - The average breeding tries needed to complete the route
+ **tests** - How many tests need to be done
+ **steps** - How many steps are needed (the minimum days needed if you complete each step in just one day)

The sorting order defaults to top from bottom on this list (power is the most important)
You can change the order, combine them cleverly or think of any other scoring.

##### Once you specify the order the programm does the rest and creates the routes you desire

Keep in mind: The algorithm only breeds flowers whose genes are known. This can get in the way of
some easy routes.
(e.g. red cosmo + yellow cosmo = orange cosmo. This orange can be one of two genes.
You can still breed this orange cosmo with itself to get a black cosmo (parent gene does not matter that much))
Black Cosmos are the only flower type where this seems to play a role.


### Some optimal paths and other results:
In `paths_simple.txt` you can find the optimal paths for all colors.\
In `hybrids_simple.txt` you can find the optimal path for all possible hybrids.

If you allow testing steps you can find even better routes in `paths.txt` and `hybrids.txt` respectively.
If you have access to the mystery island flowers you can find the best routes in `paths_island.txt` and `hybrids_island.txt`.
If you only care about short routes because you will use a massive amount of flowers anyway
you will find the shortest routes in `paths_short` and `hybrids_short`


Entries are of the form:\
`<{Genes} | {Color} | {Steps} {Power} <- {parent genes} {testing} [{chances}]>`

`{Genes}` represents the genes of a flower. for example a flower with `RR` gene with be shown with a 2 in
the slot for the `R` gene, `Rr` would be 1 `rr` 0.
Only relevant genes are given in a list with order (`R`,`Y`,`O`,`W`,`S`).
We use Paleh's notation meaning `WW` is 2 (even tough in the game code it would be 0).

`{testing}` is optional and has syntax: \
`[alt: {alt Genes} breed with {Test Genes} for {accept colors}]`

where `{alt Genes}` are the genes of the other flower variant of the same color that
can come from the same parents, `{Test Genes}` the genes of the flower to do the test with.
If your test yields a color from `{accepted colors}` it is safe to continue.

#### Testing
Testing steps are what makes breeding route so much faster than the reliance on good routes.
Many runs of the algorithm showed that there is surprisingly only on test which was useful at all:\
Test a flower variant with a seed flower. The desired flower variant can create offspring that the other can't.
The wrong flower variant is not detectable.

Simpler test where you can detect wrong or correct flowers are to weak to make a difference.
Stronger test where you can test against other flowers except seed flowers don't allow
for better and faster routes because getting other flowers as testers is too expensive.

### Further explanation and proofs:

The characteristic that is the hardest to obtain is the excepted breeding 'trys',
in other words the average of needed 'tries'.
A try in this context is what happens when two flowers make on child flower.

##### Example:
Imagine a route that consists of two steps:

+ Purple + Purple -> Yellow (25 %)
+ Yellow + Yellow -> red (12.5 %)

If you get yellow with a chance of 1:4 / 25 % than on average you need 4 tries (see Expectation of a binomial distributed random variable).
After that you get red with 1:8 / 12.5 % -  you need 8 tries average.

This means the power of this route is 4 + 8 = 12.
You could argue that you need two yellows with 4 steps each. but since you can duplicate flowers
in AC:NH you really just need 4 + 1 + 8 = 13 steps, including one duplication step.
To avoid confusion this duplication step is excluded from the power.

Getting the expected number of tries gets really complicated if you want to factor in testing.\
For a given step that breeds Flower A but also the Flower B the average tries is:\
`K * (1 + pB / pA) + 1 / pA` where `K = 1 / (pA * P(Z in tA | F = A) + pB * P(Z in tb | F = B))` is the test level\
`pA` and `pB` are the probability of the Flowers A and B, for `tA` and `tB` see below.

#### Sketch of a Proof

Let `X` be the tries needed to complete the whole step including testing\
Let `Y` be the tries needed to finish testing of one flower (we can conclude if we have the correct or incorecct flower)\
Let `Z` be the number of childs we have to test to get the correct one (e.g. `Z == 3` -> the 3th flower was correct)

We can compute the average/exceptation of these random variables:\
For a breeding step which results in the two flowers `A`, `B`
with probability `pA` and `pB` accordingly, and a test flower `t`,
let `tA` be the set of colors that `t`, `A` can breed and tB the set of colors that `t`, `tB` can breed.\
We can conclude that we have `A` from testing on `A` or `B` if and only if `tA \ tB` is not empty.\
We can finish testing if we get a child in `R := (tA \ tB) u (tB \ tA)`

Let `F` be the first flower we get that is either A or B\
Let `Z` be the color that we get if we breed the first testable flower `F`
(this could be A or B) to the test flower.\
(This obviously depends on `F`)

The probability that we can finish testing after one try:
```
P(Z in R | F = A or F = B)
= P(Z in R and (F = A or F = B)) / P(F = A or F = B)
= (P(F = A) * P(Z in R |F = A) + P(F = B) * P(Z in R|F = B)) / (pA + pB)
= (pA * P(Z in (tA \ tB) | F = A) + pB * P(Z in (tB \ tA) | F = B)) / (pA + pB)
```
Because Y has a geometric distribution we can conclude the average tries for each test(*):\
`K := E[Y] = 1 / P(Z in R | F = A or F = B)`\
The probability that a flower which is either A or B is indeed A:\
`p* := P(F2 = A | F2 = A or F2 = B) = P(F2 = A) / P(F2 = A or F2 = B) = pA / (pA + pB)`\
Therefore the average tests needed are (again Z has geometric distribution):\
`E[Z] = 1 / p* = (pA + pB) / pA`

Finally the average tries needed to complete all tests is:\
`E[Z] * E[Y] = K * (pA + pB) / pA =  K * (1 + pB / pA)`

This can be than added to the on average `1/pA` tries to get the first Flower A:\
`E[X] = K * (1 + pB / pA) + 1 / pA`

##### (*) Remark
This is a pretty big assumption that needs to be explained:\
We use the fact that Y has geometric distribution with parameter P(Z in R | F = A or F = B)
and Z has geometric distribution with parameter p*.\
This modelling means in every try a child flower of the parents is randomly chosen which then has some random offspring
with the test flower. In reality you don't use a new flower each time. The fact that the
underlying genes of the flower can't change influence the probabilities.
A more accurate modelling procedure would be:\
Randomly chose a flower. Test it until you find a result. Chose next flower.
The problem with this is that you get trapped in an infinite loop when you can only identify correct flowers
but can't detect wrong ones (e.g. Test: if a yellow can be bred it is correct).
In practice you will get a handful of flowers in a field were they wait to be tested.
One day on of them turns out correct. We can model like this:
Randomly chose one of the flowers on the field to have offspring. Generate random offspring.
An exact modelling of this would be to complicated but it is not far away from our model:
We can assume we get a certain flower in our field with the same chances that we get
a certain flower from the parents.


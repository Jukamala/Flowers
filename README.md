# Flowers
Optimal flower breding routes for Animal Crossing: New Horizons without 'uncertain' breeding steps.

Breeding flowers in Animal Crossing can be quite complicated.
Especially some habrids are hard to get by (the best example herefore is the blue roses which is notoriously difficult to bred).

Some currently used routes (see for example the excellent guide by Paleh https://docs.google.com/document/u/0/d/1ARIQCUc5YVEd01D7jtJT9EEJF45m07NXhAm4fOpNvCs/mobilebasic) 'uncertain' breeding step where you have to test offspring.

(e.g. the famous 'hybrid red' can be obtaned from orange and purple with 25 % while an 'uncool' red can be obtained with 25 % also. There is no easy way to check what you got)

For every flower and color I list an optimal(*) path to get a flower of that color with specifiable gene.
Routes don't include steps that result in two versions of the same color.
(This also means the routes are not necesarrly good for easyly obtainable flowers (e.g pink roses can be more easily obtained from red % white seed roses)
If you search for optimal ways to bred them look in https://docs.google.com/document/u/0/d/1ARIQCUc5YVEd01D7jtJT9EEJF45m07NXhAm4fOpNvCs/mobilebasic first.

####For whoever is interested:

I searched these route with a script that searched every possible breeding route using only 'certain' steps.

(*) Optimal means it has the highest harmonic mean of all possibilities on the way.
This guarantees the lowest excepted breeding 'trys':

If you get purple with chance 1:4 than you are excepted to need 4 trys (average)
If after that deeper in your route you get another flower with 1:8 you need 8 trys (average).

Together your excepted trys to complete both steps sums up to 16.
The harmonic mean of 1/4 and 1/8 is 1/16 (it's always just the inverse of excepted steps) and again maximizing this minimizes excepted trys.

TLDR: It's arguably the best metric to meassure goodness of routes.

In paths.txt you can find the optimal paths for all colors
In hybrids.txt you can find the optimal path for all possible hybrids obtainable in a 'certain' manner

Entries are of the form
`<{Genes} | {Color} | {Steps} {Excepted trys} <- {parent genes}>`

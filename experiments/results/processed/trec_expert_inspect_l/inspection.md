# Hypothesis inspection — trec (dleemiller/finecat-nli-l)

Head: `{'kind': 'hgb', 'learning_rate': 0.06, 'l2_regularization': 0.01}`  CV-train acc 0.9060. 25 hypotheses.

## Top 15 hypotheses by global permutation importance

| rank | importance | ±fold-std | hypothesis | intended |
|---|---|---|---|---|
| 1 | 0.0936 | 0.0055 | The text asks where something is located. | LOC |
| 2 | 0.0730 | 0.0094 | The text can be answered with a numeric value. | NUM |
| 3 | 0.0374 | 0.0043 | The text asks for a date, year, or period of time. | NUM |
| 4 | 0.0310 | 0.0037 | The text asks for the definition of a term. | DESC |
| 5 | 0.0198 | 0.0065 | The text asks for a description of something's meaning or purpose. | DESC |
| 6 | 0.0144 | 0.0048 | The text asks for the name of a thing or object. | ENTY |
| 7 | 0.0058 | 0.0048 | The text asks what an abbreviation or acronym stands for. | ABBR |
| 8 | 0.0044 | 0.0081 | The text asks for the name of a person. | HUM |
| 9 | 0.0042 | 0.0068 | The text asks for the identity of an individual. | HUM |
| 10 | 0.0040 | 0.0047 | The text asks about a group, team, or organization of people. | HUM |
| 11 | 0.0020 | 0.0026 | The text asks how many of something there are. | NUM |
| 12 | 0.0014 | 0.0036 | The text asks who did something or who is responsible. | HUM |
| 13 | 0.0006 | 0.0111 | The text asks for an explanation of why something happens. | DESC |
| 14 | 0.0006 | 0.0021 | The text asks for a distance, size, or measurement. | NUM |
| 15 | 0.0002 | 0.0047 | The text asks which animal, plant, or substance is being described. | ENTY |

## Top hypotheses per class (mean P(entail) on that class's test texts)

**ABBR**
- 0.993  The text asks for the definition of a term.  _(→DESC)_
- 0.992  The text asks what an abbreviation or acronym stands for.  _(→ABBR)_
- 0.985  The text asks what something is called.  _(→ENTY)_
- 0.984  The question is about what a set of letters means.  _(→ABBR)_
- 0.952  The text asks for the full form of an initialism.  _(→ABBR)_

**ENTY**
- 0.867  The text asks for the name of a thing or object.  _(→ENTY)_
- 0.809  The text asks what something is called.  _(→ENTY)_
- 0.662  The text asks for the definition of a term.  _(→DESC)_
- 0.289  The text asks for a description of something's meaning or purpose.  _(→DESC)_
- 0.261  The text asks for the full form of an initialism.  _(→ABBR)_

**DESC**
- 0.958  The text asks for the definition of a term.  _(→DESC)_
- 0.941  The text asks for a description of something's meaning or purpose.  _(→DESC)_
- 0.923  The text asks what something is called.  _(→ENTY)_
- 0.919  The text asks what an abbreviation or acronym stands for.  _(→ABBR)_
- 0.913  The text asks for the name of a thing or object.  _(→ENTY)_

**HUM**
- 0.930  The text asks for the identity of an individual.  _(→HUM)_
- 0.920  The text asks for the name of a person.  _(→HUM)_
- 0.904  The text asks for the name of a thing or object.  _(→ENTY)_
- 0.618  The text asks who did something or who is responsible.  _(→HUM)_
- 0.577  The text asks for the full form of an initialism.  _(→ABBR)_

**LOC**
- 0.971  The text asks where something is located.  _(→LOC)_
- 0.887  The text asks about a geographic location.  _(→LOC)_
- 0.861  The text asks which region or area something is in.  _(→LOC)_
- 0.828  The text asks for the name of a thing or object.  _(→ENTY)_
- 0.773  The text asks for the name of a place, city, or country.  _(→LOC)_

**NUM**
- 0.561  The text can be answered with a numeric value.  _(→NUM)_
- 0.538  The text asks for a number or a count.  _(→NUM)_
- 0.462  The text asks for a date, year, or period of time.  _(→NUM)_
- 0.304  The text asks for the full form of an initialism.  _(→ABBR)_
- 0.226  The text asks for the definition of a term.  _(→DESC)_

## Redundant hypothesis pairs (|corr| ≥ 0.9)

- corr +0.92: “The text asks for the name of a person.” ⟷ “The text asks for the identity of an individual.”

## Per-hypothesis exemplars (top by importance)

**The text asks where something is located.**
- entailed by: “Where is the Little League Museum ?” | “Where is the Grand Canyon ?” | “Where is the Shawnee National Forest ?”
- contradicted by: “What primary colors do you mix to make orange ?” | “What is the colorful Korean traditional dress called ?” | “How fast is sound ?”

**The text can be answered with a numeric value.**
- entailed by: “How old was Elvis Presley when he died ?” | “Mexican pesos are worth what in U.S. dollars ?” | “How often does Old Faithful erupt at Yellowstone National Park ?”
- contradicted by: “Who was the abolitionist who led the raid on Harper 's Ferry in 1859 ?” | “What is pilates ?” | “Who was the first female United States Representative ?”

**The text asks for a date, year, or period of time.**
- entailed by: “What year did WWII begin ?” | “What year did the NFL go on strike ?” | “What year did Canada join the United Nations ?”
- contradicted by: “What currency do they use in Brazil ?” | “What primary colors do you mix to make orange ?” | “What currency does Argentina use ?”

**The text asks for the definition of a term.**
- entailed by: “What is genocide ?” | “What is mad cow disease ?” | “What is influenza ?”
- contradicted by: “Who is the only president to serve 2 non-consecutive terms ?” | “What year did the NFL go on strike ?” | “What day and month did John Lennon die ?”

**The text asks for a description of something's meaning or purpose.**
- entailed by: “What does NASA stand for ?” | “What does a defibrillator do ?” | “What does CPR stand for ?”
- contradicted by: “What day and month did John Lennon die ?” | “What state is the geographic center of the lower 48 states ?” | “What is the melting point of gold ?”

**The text asks for the name of a thing or object.**
- entailed by: “What is a baby turkey called ?” | “What breed of hunting dog did the Beverly Hillbillies own ?” | “What is a baby lion called ?”
- contradicted by: “How many feet in a mile ?” | “How fast is alcohol absorbed ?” | “How fast is sound ?”

**The text asks what an abbreviation or acronym stands for.**
- entailed by: “What does the acronym NASA stand for ?” | “What does CPR stand for ?” | “What does USPS stand for ?”
- contradicted by: “What monastery was raided by Vikings in the late eighth century ?” | “How old was Joan of Arc when she died ?” | “What year did the Titanic start on its journey ?”

**The text asks for the name of a person.**
- entailed by: “Who was Abraham Lincoln ?” | “Who is the congressman from state of Texas on the armed forces committee ?” | “Who was the first man to fly across the Pacific Ocean ?”
- contradicted by: “What gasses are in the troposphere ?” | “What river flows between Fargo , North Dakota and Moorhead , Minnesota ?” | “What continent is Argentina on ?”

**The text asks for the identity of an individual.**
- entailed by: “Who is the congressman from state of Texas on the armed forces committee ?” | “Who is the governor of Alaska ?” | “Who was Abraham Lincoln ?”
- contradicted by: “What currency does Argentina use ?” | “What currency does Luxembourg use ?” | “How many gallons of water are there in a cubic foot ?”

**The text asks about a group, team, or organization of people.**
- entailed by: “What was the last year that the Chicago Cubs won the World Series ?” | “What year did the NFL go on strike ?” | “What baseball team was the first to make numbers part of their uniform ?”
- contradicted by: “Material called linen is made from what plant ?” | “What precious stone is a form of pure carbon ?” | “What do you call a newborn kangaroo ?”

**The text asks how many of something there are.**
- entailed by: “How many Great Lakes are there ?” | “How many Admirals are there in the U.S. Navy ?” | “What is the population of Australia ?”
- contradicted by: “What year did Mussolini seize power in Italy ?” | “What monastery was raided by Vikings in the late eighth century ?” | “What country did Ponce de Leon come from ?”

**The text asks who did something or who is responsible.**
- entailed by: “Who painted the ceiling of the Sistine Chapel ?” | “Who was the abolitionist who led the raid on Harper 's Ferry in 1859 ?” | “Who invented the slinky ?”
- contradicted by: “How fast is alcohol absorbed ?” | “How much fiber should you have per day ?” | “How far is it from Denver to Aspen ?”

**The text asks for an explanation of why something happens.**
- entailed by: “Why in tennis are zero points called love ?” | “Why does the moon turn orange ?” | “Why is the sun yellow ?”
- contradicted by: “What soviet seaport is on the Black Sea ?” | “How far away is the moon ?” | “What is the street address of the White House ?”

**The text asks for a distance, size, or measurement.**
- entailed by: “How many feet in a mile ?” | “How tall is the Sears Building ?” | “How wide is the Milky Way galaxy ?”
- contradicted by: “What primary colors do you mix to make orange ?” | “What breed of hunting dog did the Beverly Hillbillies own ?” | “What color is indigo ?”

**The text asks which animal, plant, or substance is being described.**
- entailed by: “What is Maryland 's state bird ?” | “What is New York 's state bird ?” | “What are amphibians ?”
- contradicted by: “In Poland , where do most people live ?” | “What year did Mussolini seize power in Italy ?” | “In what spacecraft did U.S. astronaut Alan Shepard make his historic 1961 flight ?”

## Error cases with top-activating hypotheses

- [true **ENTY** → pred **LOC**] “What is the major fault line near Kentucky ?”
  top: The text asks what something is called.; The text asks where something is located.; The text asks for the name of a thing or object.
- [true **DESC** → pred **LOC**] “What is the Milky Way ?”
  top: The text asks what something is called.; The text asks for the name of a thing or object.; The text asks for the definition of a term.
- [true **ENTY** → pred **NUM**] “What is the electrical output in Madrid , Spain ?”
  top: The text asks for a number or a count.; The text asks which region or area something is in.; The text asks where something is located.
- [true **LOC** → pred **DESC**] “What are the twin cities ?”
  top: The text asks what something is called.; The text asks for the definition of a term.; The text asks for the name of a thing or object.
- [true **HUM** → pred **LOC**] “What is the oldest university in the US ?”
  top: The text asks for the name of a thing or object.; The text asks where something is located.; The text asks what something is called.
- [true **ENTY** → pred **DESC**] “What do you call a professional map drawer ?”
  top: The text asks what something is called.; The text asks for the name of a thing or object.; The text asks for the definition of a term.
- [true **ENTY** → pred **LOC**] “What is the source of natural gas ?”
  top: The text asks for the definition of a term.; The text asks where something is located.; The text asks for the name of a thing or object.
- [true **ENTY** → pred **DESC**] “What is foot and mouth disease ?”
  top: The text asks for the definition of a term.; The text asks what something is called.; The text asks for the name of a thing or object.
- [true **DESC** → pred **LOC**] “What is the Moulin Rouge ?”
  top: The text asks for the name of a thing or object.; The text asks what something is called.; The text asks for the definition of a term.
- [true **HUM** → pred **LOC**] “What chain store is headquartered in Bentonville , Arkansas ?”
  top: The text asks where something is located.; The text asks for the name of a place, city, or country.; The text asks for the name of a thing or object.

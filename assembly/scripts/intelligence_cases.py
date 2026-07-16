"""Deterministically graded, original intelligence-evaluation cases.

The task families follow public benchmark designs, but the individual questions
are original variants so the suite measures reasoning instead of recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json


@dataclass(frozen=True)
class Case:
    id: int
    difficulty: str
    category: str
    source: str
    prompt: str
    expected: str
    grader: str = "exact"


def build_cases() -> list[Case]:
    cases: list[Case] = []

    def add(
        difficulty: str,
        category: str,
        source: str,
        prompt: str,
        expected: str,
        grader: str = "exact",
    ) -> None:
        cases.append(
            Case(len(cases) + 1, difficulty, category, source, prompt, expected, grader)
        )

    # 1-40: IFEval-style deterministic instruction following.
    instruction_cases = [
        ("Convert `Quiet Rivers Flow` to lowercase. Reply exactly with the converted text.", "quiet rivers flow"),
        ("Convert `silver moon` to uppercase. Reply exactly with the converted text.", "SILVER MOON"),
        ("Reverse the word order in `red green blue`. Reply exactly with words separated by one space.", "blue green red"),
        ("Alphabetize `pear apple plum banana`. Reply exactly as lowercase words separated by commas with no spaces.", "apple,banana,pear,plum"),
        ("Sort 14, 3, 9, 1 ascending. Reply exactly as comma-separated numbers with no spaces.", "1,3,9,14"),
        ("Sort 5, 12, 2, 8 descending. Reply exactly as comma-separated numbers with no spaces.", "12,8,5,2"),
        ("Remove duplicate words from `ant bee ant cat bee` while preserving first appearance. Reply exactly with one space between words.", "ant bee cat"),
        ("Replace every hyphen in `north-east-south` with `/`. Reply exactly with the result.", "north/east/south"),
        ("Return valid compact JSON mapping A to 2 and B to 7. Use this exact key order and no spaces.", '{"A":2,"B":7}', "json"),
        ("Return a valid compact JSON array containing the even numbers from 1,2,3,4,5,6 in order. No spaces.", "[2,4,6]", "json"),
        ("Write exactly three bullet lines using `- `: `Fast`, `Clear`, `Safe`, in that order. No other text.", "- Fast\n- Clear\n- Safe"),
        ("Join `alpha`, `beta`, and `gamma` using semicolons. Reply exactly, with no spaces.", "alpha;beta;gamma"),
        ("Convert the date July 4, 2026 to ISO YYYY-MM-DD. Reply exactly with the date.", "2026-07-04"),
        ("Extract the initials of `National Aeronautics Space Agency`. Reply exactly in uppercase with no punctuation.", "NASA"),
        ("Take the first character of each word in `bright old lantern dawn`. Reply exactly in lowercase.", "bold"),
        ("Take the last character of each word in `rain echo lamp`. Reply exactly with no separators.", "nop"),
        ("Count the words in `small steps create steady progress`. Reply exactly with one digit.", "5"),
        ("Count lowercase letter `a` in `bananas and papayas`. Reply exactly with one digit.", "7"),
        ("Remove all spaces from `a b  c   d`. Reply exactly.", "abcd"),
        ("Keep only ASCII digits from `a1-b2_c3`. Reply exactly.", "123"),
        ("Format name=Nora and score=91 exactly as `name|score`, substituting values. No other text.", "Nora|91"),
        ("Wrap the word `core` in exactly two leading and two trailing asterisks. Reply exactly.", "**core**"),
        ("Place `left`, `middle`, `right` on separate lines in that order. No bullets or other text.", "left\nmiddle\nright"),
        ("Reply with exactly the five words `calm minds solve hard problems` and nothing else.", "calm minds solve hard problems"),
        ("Change only the first letter of `python` to uppercase. Reply exactly.", "Python"),
        ("Rotate `A B C D` left by one position. Reply exactly with single spaces.", "B C D A"),
        ("Rotate `1 2 3 4 5` right by two positions. Reply exactly with single spaces.", "4 5 1 2 3"),
        ("Interleave `A B C` with `1 2 3`, starting with a letter. Reply exactly with single spaces.", "A 1 B 2 C 3"),
        ("Convert binary `101101` to decimal. Reply exactly with the number.", "45"),
        ("Convert decimal 26 to lowercase hexadecimal without a prefix. Reply exactly.", "1a"),
        ("Write the plural of `analysis`. Reply with exactly one lowercase word.", "analyses"),
        ("Choose the correctly spelled word: A) accomodate B) accommodate C) acommodate. Reply exactly with the letter.", "B"),
        ("Return the unique letters of `mississippi` in first-appearance order. Reply exactly with no separators.", "misp"),
        ("Apply ROT13 to lowercase `hello`. Reply exactly in lowercase.", "uryyb"),
        ("Compute the length of string `assembly` in characters. Reply exactly with the number.", "8"),
        ("Swap the two comma-separated fields in `east,west`. Reply exactly with a comma and no spaces.", "west,east"),
        ("Convert `one two three` to `one|two|three`. Reply exactly.", "one|two|three"),
        ("Return valid compact JSON with key `ok` and boolean value true. No spaces or other text.", '{"ok":true}', "json"),
        ("Write exactly two numbered lines: first `1. Plan`, then `2. Build`. No other text.", "1. Plan\n2. Build"),
        ("Answer with exactly `YES` if 18 is divisible by 3, otherwise exactly `NO`.", "YES"),
    ]
    for prompt, expected, *grader in instruction_cases:
        add("easy", "instruction-following", "IFEval-inspired original", prompt, expected, grader[0] if grader else "exact")

    # 41-80: GSM8K-style arithmetic and quantitative reasoning.
    arithmetic = [
        ("Compute 37 + 58. Reply exactly with the number.", "95"),
        ("Compute 144 - 79. Reply exactly with the number.", "65"),
        ("Compute 23 * 17. Reply exactly with the number.", "391"),
        ("Compute 936 / 12. Reply exactly with the number.", "78"),
        ("Compute 18 + 6 * 7 using standard precedence. Reply exactly with the number.", "60"),
        ("Compute (45 - 13) * 3. Reply exactly with the number.", "96"),
        ("Compute 2^8. Reply exactly with the number.", "256"),
        ("Compute the greatest common divisor of 84 and 126. Reply exactly with the number.", "42"),
        ("Compute the least common multiple of 12 and 18. Reply exactly with the number.", "36"),
        ("Compute 15% of 240. Reply exactly with the number.", "36"),
        ("Solve 4x + 7 = 31. Reply exactly with x.", "6"),
        ("Solve 5(x - 3) = 40. Reply exactly with x.", "11"),
        ("Solve 3x - 8 = 2x + 9. Reply exactly with x.", "17"),
        ("Solve x/6 + 4 = 9. Reply exactly with x.", "30"),
        ("Solve 2(x + 5) - 3 = 21. Reply exactly with x.", "7"),
        ("If x+y=18 and x-y=4, reply exactly as `x,y`.", "11,7"),
        ("Solve x^2 = 121 for the positive value of x. Reply exactly with x.", "11"),
        ("An arithmetic sequence starts 7, 12, 17. What is its 10th term? Reply exactly with the number.", "52"),
        ("A geometric sequence starts 3, 6, 12. What is its 7th term? Reply exactly with the number.", "192"),
        ("What is the sum of integers 1 through 20? Reply exactly with the number.", "210"),
        ("Mina has 18 pencils, buys 7 more, then gives away 9. How many remain? Reply exactly with the number.", "16"),
        ("Six boxes each hold 14 bolts. Nine bolts are used. How many remain? Reply exactly with the number.", "75"),
        ("A train travels 180 km in 3 hours at constant speed. What is its speed in km/h? Reply exactly with the number.", "60"),
        ("A $80 jacket is discounted by 25%. What is the sale price in dollars? Reply exactly with the number.", "60"),
        ("A recipe uses 3 cups for 8 servings. How many cups for 20 servings? Reply exactly with the decimal number.", "7.5"),
        ("A rectangle is 13 cm by 8 cm. What is its area in square cm? Reply exactly with the number.", "104"),
        ("A square has perimeter 44 cm. What is its area in square cm? Reply exactly with the number.", "121"),
        ("A tank contains 120 liters and loses 15 liters per hour. How many liters remain after 6 hours? Reply exactly with the number.", "30"),
        ("Three friends split $84 equally, then each spends $9. How many dollars does each have left? Reply exactly with the number.", "19"),
        ("A class has 12 girls and 18 boys. What percent of the class are girls? Reply exactly with the percentage including `%`.", "40%"),
        ("A fair coin is flipped twice. Probability of exactly one head? Reply exactly as a simplified fraction.", "1/2"),
        ("Two fair dice are rolled. Probability the sum is 7? Reply exactly as a simplified fraction.", "1/6"),
        ("A bag has 3 red and 2 blue balls. Probability of red on one draw? Reply exactly as a simplified fraction.", "3/5"),
        ("Choose 2 people from 5. How many unordered pairs are possible? Reply exactly with the number.", "10"),
        ("What is 3/4 + 5/8? Reply exactly as a simplified fraction.", "11/8"),
        ("What is 7/9 - 2/3? Reply exactly as a simplified fraction.", "1/9"),
        ("What is (5/6) * (9/10)? Reply exactly as a simplified fraction.", "3/4"),
        ("What is (4/5) / (2/3)? Reply exactly as a simplified fraction.", "6/5"),
        ("The mean of 6, 8, 10, and 16 is what? Reply exactly with the number.", "10"),
        ("A value rises from 50 to 65. What is the percent increase? Reply exactly with the percentage including `%`.", "30%"),
    ]
    for i, (prompt, expected) in enumerate(arithmetic):
        add("easy" if i < 10 else "medium", "quantitative-reasoning", "GSM8K-inspired original", prompt, expected)

    # 81-120: BBH-style logical and multistep reasoning.
    reasoning = [
        ("Find the next term: 1, 4, 9, 16, 25. Reply exactly with the number.", "36"),
        ("Find the next term: 3, 7, 15, 31. Reply exactly with the number.", "63"),
        ("Find the next term: 100, 99, 95, 86, 70. Reply exactly with the number.", "45"),
        ("Find the next term: 2, 3, 5, 8, 12, 17. Reply exactly with the number.", "23"),
        ("Find the next term: 81, 27, 9, 3. Reply exactly with the number.", "1"),
        ("Find the next letter: A, C, F, J, O. Reply exactly with the uppercase letter.", "U"),
        ("Find the next term: 1, 2, 6, 24, 120. Reply exactly with the number.", "720"),
        ("Find the missing term: 5, 10, 20, ?, 80. Reply exactly with the number.", "40"),
        ("Find the next term: 14, 13, 11, 8, 4. Reply exactly with the number.", "-1"),
        ("Find the next term: 2, 5, 11, 23, 47. Reply exactly with the number.", "95"),
        ("All nims are lats. All lats are zogs. Must all nims be zogs? Reply exactly YES or NO.", "YES"),
        ("No pels are tars. Some wims are pels. Can those wims be tars? Reply exactly YES or NO.", "NO"),
        ("Some artists are pilots. All pilots are readers. Must some artists be readers? Reply exactly YES or NO.", "YES"),
        ("All roses are flowers. Some flowers fade quickly. Must some roses fade quickly? Reply exactly YES or NO.", "NO"),
        ("No metal objects are transparent. This object is transparent. Can it be metal? Reply exactly YES or NO.", "NO"),
        ("Every dax is either red or blue. No dax is blue. Must every dax be red? Reply exactly YES or NO.", "YES"),
        ("Some kets are not round. All kets are wooden. Must some wooden things not be round? Reply exactly YES or NO.", "YES"),
        ("All doctors studied biology. Lee studied biology. Must Lee be a doctor? Reply exactly YES or NO.", "NO"),
        ("No cats are reptiles. Some pets are cats. Must some pets not be reptiles? Reply exactly YES or NO.", "YES"),
        ("All A are B. No B are C. Some D are C. Must any D be A? Reply exactly YES or NO.", "NO"),
        ("If today is Monday, what day is 45 days later? Reply exactly with the weekday.", "Thursday"),
        ("If today is Friday, what day was 10 days ago? Reply exactly with the weekday.", "Tuesday"),
        ("A meeting starts at 13:45 and lasts 95 minutes. Reply exactly with the end time in 24-hour HH:MM format.", "15:20"),
        ("A clock shows 23:30. What time is it 150 minutes later? Reply exactly in 24-hour HH:MM format.", "02:00"),
        ("If March 1 is a Wednesday, what weekday is March 15? Reply exactly with the weekday.", "Wednesday"),
        ("Tasks require A before C, B before C, and C before D. Is `B A C D` valid? Reply exactly YES or NO.", "YES"),
        ("Tasks require P before Q and Q before R. Is `Q P R` valid? Reply exactly YES or NO.", "NO"),
        ("Tasks require A before D, B before D, and C before E. Is `C B A E D` valid? Reply exactly YES or NO.", "YES"),
        ("Tasks require X before Y and Z before X. Which must be first among X,Y,Z? Reply exactly with the letter.", "Z"),
        ("Tasks require L before M, M before N, and K before N. Can N be first? Reply exactly YES or NO.", "NO"),
        ("Ada is taller than Ben. Ben is taller than Cy. Who is shortest? Reply exactly with the name.", "Cy"),
        ("Lia finished before Moe but after Nia. Who finished first? Reply exactly with the name.", "Nia"),
        ("Ava faces north, turns right, then right, then left. Which direction does she face? Reply exactly with the lowercase direction.", "east"),
        ("A cube has all faces painted and is cut into 27 equal smaller cubes. How many small cubes have exactly three painted faces? Reply exactly with the number.", "8"),
        ("Three boxes are all mislabeled `Apples`, `Oranges`, `Mixed`. Which labeled box should be sampled first? Reply exactly with its label.", "Mixed"),
        ("A says `B is lying`. B says `A and I are different types`. Truth-tellers always tell truth; liars always lie. What is A? Reply exactly `truth-teller` or `liar`.", "liar"),
        ("There are three switches downstairs and one controls an upstairs bulb. You may go upstairs once. Which physical property besides light reveals the switch? Reply exactly with one lowercase word.", "heat"),
        ("A farmer has 17 sheep and all but 9 leave. How many remain? Reply exactly with the number.", "9"),
        ("You pass the runner in second place. What place are you now? Reply exactly with the ordinal word in lowercase.", "second"),
        ("Two fathers and two sons share three apples equally, one each. What is the minimum number of people? Reply exactly with the number.", "3"),
    ]
    for i, (prompt, expected) in enumerate(reasoning):
        add("medium" if i < 30 else "hard", "logical-reasoning", "BBH-inspired original", prompt, expected)

    # 121-150: MMLU-style broad knowledge, all with exact option grading.
    knowledge = [
        ("Which organelle is the main site of ATP production in eukaryotic cells? A) Nucleus B) Mitochondrion C) Ribosome D) Golgi apparatus. Reply exactly with the letter.", "B"),
        ("Which gas is most abundant in Earth's atmosphere? A) Oxygen B) Carbon dioxide C) Nitrogen D) Argon. Reply exactly with the letter.", "C"),
        ("What is the SI unit of electric current? A) Volt B) Watt C) Ohm D) Ampere. Reply exactly with the letter.", "D"),
        ("Which particle has a negative electric charge? A) Proton B) Neutron C) Electron D) Photon. Reply exactly with the letter.", "C"),
        ("A solution with pH 3 is: A) Acidic B) Neutral C) Basic D) Radioactive. Reply exactly with the letter.", "A"),
        ("Which process converts atmospheric nitrogen into biologically usable compounds? A) Respiration B) Nitrogen fixation C) Transpiration D) Fermentation. Reply exactly with the letter.", "B"),
        ("Which law states that pressure and volume are inversely related at constant temperature? A) Boyle's law B) Charles's law C) Ohm's law D) Hooke's law. Reply exactly with the letter.", "A"),
        ("What type of bond shares electron pairs? A) Ionic B) Metallic C) Covalent D) Hydrogen. Reply exactly with the letter.", "C"),
        ("Which blood cells primarily transport oxygen? A) Platelets B) Red blood cells C) Neurons D) White blood cells. Reply exactly with the letter.", "B"),
        ("In genetics, different forms of the same gene are called: A) Alleles B) Chromatids C) Ribosomes D) Phenotypes. Reply exactly with the letter.", "A"),
        ("Who wrote `Pride and Prejudice`? A) George Eliot B) Jane Austen C) Mary Shelley D) Virginia Woolf. Reply exactly with the letter.", "B"),
        ("The Magna Carta was originally issued in which country? A) France B) Spain C) England D) Italy. Reply exactly with the letter.", "C"),
        ("Which civilization built Machu Picchu? A) Maya B) Inca C) Roman D) Egyptian. Reply exactly with the letter.", "B"),
        ("The Renaissance began primarily in: A) Italy B) Norway C) Russia D) Canada. Reply exactly with the letter.", "A"),
        ("Which ocean is the largest? A) Atlantic B) Indian C) Arctic D) Pacific. Reply exactly with the letter.", "D"),
        ("What is the capital of Australia? A) Sydney B) Melbourne C) Canberra D) Perth. Reply exactly with the letter.", "C"),
        ("The longest river in South America is the: A) Nile B) Amazon C) Danube D) Yangtze. Reply exactly with the letter.", "B"),
        ("Which language family includes Spanish? A) Romance B) Germanic C) Slavic D) Uralic. Reply exactly with the letter.", "A"),
        ("In economics, opportunity cost is: A) A tax rate B) The next-best alternative forgone C) Total revenue D) A sunk cost only. Reply exactly with the letter.", "B"),
        ("Inflation means: A) A sustained rise in the general price level B) Falling output only C) Fixed exchange rates D) Zero unemployment. Reply exactly with the letter.", "A"),
        ("Which data structure uses FIFO order? A) Stack B) Queue C) Heap D) Tree. Reply exactly with the letter.", "B"),
        ("Binary search requires data to be: A) Random B) Encrypted C) Sorted D) Duplicated. Reply exactly with the letter.", "C"),
        ("Which protocol normally maps domain names to IP addresses? A) DNS B) FTP C) SMTP D) SSH. Reply exactly with the letter.", "A"),
        ("In object-oriented programming, bundling data with methods is: A) Recursion B) Encapsulation C) Compilation D) Serialization. Reply exactly with the letter.", "B"),
        ("Which normal form removes partial dependencies on a composite key? A) 1NF B) 2NF C) 3NF D) BCNF only. Reply exactly with the letter.", "B"),
        ("A deductively valid argument guarantees that if premises are true, the conclusion is: A) Popular B) True C) Novel D) Uncertain. Reply exactly with the letter.", "B"),
        ("Utilitarianism primarily evaluates actions by their: A) Consequences B) Age C) Grammar D) Legal citation. Reply exactly with the letter.", "A"),
        ("In psychology, classical conditioning is most associated with: A) Pavlov B) Maslow C) Piaget D) Chomsky. Reply exactly with the letter.", "A"),
        ("Which branch of government typically interprets laws? A) Executive B) Legislative C) Judicial D) Municipal. Reply exactly with the letter.", "C"),
        ("A market with one seller is called: A) Oligopoly B) Monopoly C) Monopsony D) Perfect competition. Reply exactly with the letter.", "B"),
    ]
    for prompt, expected in knowledge:
        add("medium", "knowledge", "MMLU-inspired original", prompt, expected)

    # 151-170: TruthfulQA-style resistance to common misconceptions.
    truthful = [
        ("Which is accurate? A) Humans use only 10% of their brains B) Humans use many brain regions over time. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Lightning never strikes the same place twice B) Lightning can strike the same place repeatedly. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) The Great Wall is plainly visible from the Moon unaided B) It is generally not visible from the Moon unaided. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Seasons are caused mainly by Earth's axial tilt B) Seasons are caused mainly by changing Earth-Sun distance. Reply exactly with the letter.", "A"),
        ("Which is accurate? A) Antibiotics treat viral infections directly B) Antibiotics target bacteria, not viruses. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Goldfish have only a three-second memory B) Goldfish can retain memories much longer. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Bats are blind B) Bats can see, though many also echolocate. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Sugar reliably makes all children hyperactive B) controlled studies do not support that broad claim. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Cracking knuckles is proven to cause arthritis B) evidence does not show it causes arthritis. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Vikings routinely wore horned helmets B) evidence does not support horned helmets as normal Viking wear. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Water always drains in opposite swirl directions across hemispheres B) household drain direction is dominated by container conditions. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) A penny dropped from a skyscraper reaches lethal bullet-like speed B) air resistance limits its speed far below bullet speed. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Swallowed gum normally remains for seven years B) it generally passes through the digestive system. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Chameleons change color only to match backgrounds B) they change color for signaling and physiological regulation too. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Deoxygenated human blood is blue B) human blood remains red, though veins can appear blue. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Shaving makes hair grow back thicker B) shaving does not change follicle thickness. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Medieval Europeans universally believed Earth was flat B) educated medieval Europeans generally knew Earth was spherical. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Toilets flush in a hemisphere-determined direction B) toilet design and jets dominate flushing direction. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) Evolution says humans descended from modern monkeys B) humans and modern monkeys share ancestors. Reply exactly with the letter.", "B"),
        ("Which is accurate? A) There is a dark side of the Moon that never gets sunlight B) all lunar regions receive sunlight over time except some polar craters. Reply exactly with the letter.", "B"),
    ]
    for prompt, expected in truthful:
        add("medium", "truthfulness", "TruthfulQA-inspired original", prompt, expected)

    # 171-200: HumanEval-style code understanding without executing model code.
    coding = [
        ("Python: `print(sum([2, 4, 6]))`. What is printed? Reply exactly with the output.", "12"),
        ("Python: `x=[1,2]; y=x; y.append(3); print(len(x))`. What is printed? Reply exactly with the output.", "3"),
        ("Python: `print('agent'[::-1])`. What is printed? Reply exactly with the output.", "tnega"),
        ("Python: `print([x*x for x in range(4)])`. What is printed? Reply exactly with the output.", "[0, 1, 4, 9]"),
        ("Python: `d={'a':1}; print(d.get('b',5))`. What is printed? Reply exactly with the output.", "5"),
        ("Python: `print(7 // 2, 7 % 2)`. What is printed? Reply exactly with the output.", "3 1"),
        ("Python: `print(bool([]), bool([0]))`. What is printed? Reply exactly with the output.", "False True"),
        ("Python: `a=[3,1,2]; print(sorted(a)[0], a[0])`. What is printed? Reply exactly with the output.", "1 3"),
        ("Python: `print(','.join(['a','b','c']))`. What is printed? Reply exactly with the output.", "a,b,c"),
        ("Python: `print(len(set([1,1,2,3,3])))`. What is printed? Reply exactly with the output.", "3"),
        ("JavaScript: `console.log([1,2,3].map(x => x * 2).join(','))`. What is printed? Reply exactly with the output.", "2,4,6"),
        ("JavaScript: `console.log(typeof null)`. What is printed? Reply exactly with the output.", "object"),
        ("JavaScript: `console.log('5' + 2)`. What is printed? Reply exactly with the output.", "52"),
        ("JavaScript: `console.log([3,1,2].sort().join(''))`. What is printed? Reply exactly with the output.", "123"),
        ("JavaScript: `console.log(Boolean(''))`. What is printed? Reply exactly with the output.", "false"),
        ("Which structure is best for FIFO processing? A) Stack B) Queue C) Set D) Tree. Reply exactly with the letter.", "B"),
        ("Which traversal uses an explicit queue? A) BFS B) DFS C) Binary search D) Quicksort. Reply exactly with the letter.", "A"),
        ("Average lookup time in a well-designed hash table is: A) O(1) B) O(log n) C) O(n) D) O(n^2). Reply exactly with the letter.", "A"),
        ("Worst-case time for merge sort is: A) O(1) B) O(log n) C) O(n log n) D) O(n^2). Reply exactly with the letter.", "C"),
        ("A stable sort preserves: A) Array length only B) Relative order of equal keys C) Memory addresses D) Reverse order. Reply exactly with the letter.", "B"),
        ("Which SQL clause filters groups after aggregation? A) WHERE B) HAVING C) ORDER BY D) JOIN. Reply exactly with the letter.", "B"),
        ("Which SQL expression counts rows? A) SUM(ROW) B) COUNT(*) C) SIZE(*) D) ROWS(). Reply exactly with the letter.", "B"),
        ("In Git, which command creates a new branch and switches to it? A) git merge B) git switch -c C) git fetch D) git diff. Reply exactly with the letter.", "B"),
        ("An off-by-one error most commonly involves: A) Loop boundaries B) Network encryption C) File permissions D) Color spaces. Reply exactly with the letter.", "A"),
        ("A function calling itself uses: A) Memoization B) Recursion C) Serialization D) Tokenization. Reply exactly with the letter.", "B"),
        ("For sorted input, binary search is O(log n) because each step: A) Doubles values B) Halves the search space C) Sorts again D) Checks every item. Reply exactly with the letter.", "B"),
        ("A race condition occurs when correctness depends on: A) Font choice B) Timing of concurrent operations C) Disk size only D) Variable naming. Reply exactly with the letter.", "B"),
        ("Which HTTP status class represents client errors? A) 1xx B) 2xx C) 4xx D) 5xx. Reply exactly with the letter.", "C"),
        ("What does JSON require around object keys? A) Parentheses B) Double quotes C) Backticks D) No delimiters. Reply exactly with the letter.", "B"),
        ("Which test checks a single function in isolation? A) Unit test B) Load test C) Usability test D) Smoke alarm. Reply exactly with the letter.", "A"),
    ]
    for i, (prompt, expected) in enumerate(coding):
        add("medium" if i < 15 else "hard", "code-reasoning", "HumanEval-inspired original", prompt, expected)

    # 201-360: additional IFEval-style transformations and exact formatting.
    colors = [
        "red", "blue", "green", "amber", "violet", "silver", "gold", "black", "white", "coral",
        "teal", "navy", "lime", "pink", "gray", "indigo", "bronze", "cream", "cyan", "maroon",
    ]
    animals = [
        "ant", "bear", "cat", "dog", "eagle", "fox", "goat", "hare", "ibis", "jay",
        "koala", "lynx", "mole", "newt", "otter", "panda", "quail", "raven", "seal", "tiger",
    ]
    for i in range(20):
        phrase = f"Signal {colors[i].title()} {animals[i].title()}"
        add("easy", "instruction-following", "IFEval-inspired generated", f"Convert `{phrase}` to lowercase. Reply exactly with the result.", phrase.lower())
    for i in range(20):
        phrase = f"quiet {colors[i]} {animals[(i + 3) % 20]}"
        add("easy", "instruction-following", "IFEval-inspired generated", f"Convert `{phrase}` to uppercase. Reply exactly with the result.", phrase.upper())
    for i in range(20):
        words = [colors[i], animals[i], colors[(i + 7) % 20], animals[(i + 11) % 20]]
        add("easy", "instruction-following", "IFEval-inspired generated", f"Reverse the word order in `{' '.join(words)}`. Reply exactly with single spaces.", " ".join(reversed(words)))
    for i in range(20):
        words = [animals[i], colors[(i + 5) % 20], animals[(i + 9) % 20], colors[(i + 13) % 20]]
        add("easy", "instruction-following", "IFEval-inspired generated", f"Alphabetize these lowercase words: {', '.join(words)}. Reply exactly comma-separated with no spaces.", ",".join(sorted(words)))
    for i in range(20):
        values = [i + 1, i + 4, i + 7, i + 10, i + 13]
        shift = (i % 4) + 1
        rotated = values[-shift:] + values[:-shift]
        add("medium", "instruction-following", "IFEval-inspired generated", f"Rotate `{' '.join(map(str, values))}` right by {shift} positions. Reply exactly with single spaces.", " ".join(map(str, rotated)))
    for i in range(20):
        text = f"x{i + 2}y{i + 7:02d}z{i + 13}"
        expected = "".join(char for char in text if char.isdigit())
        add("easy", "instruction-following", "IFEval-inspired generated", f"Keep only ASCII digits from `{text}`. Reply exactly with no separators.", expected)
    for i in range(20):
        expected_obj = {f"k{i}": i + 3, f"v{i}": i * 2 + 1}
        expected = json.dumps(expected_obj, separators=(",", ":"))
        add("medium", "instruction-following", "IFEval-inspired generated", f"Return compact JSON mapping `k{i}` to {i + 3} and `v{i}` to {i * 2 + 1}, in that key order. No spaces or other text.", expected, "json")
    separators = ["|", ";", "/", ":"]
    for i in range(20):
        parts = [colors[i], animals[(i + 2) % 20], str(i + 10)]
        sep = separators[i % len(separators)]
        add("easy", "instruction-following", "IFEval-inspired generated", f"Join `{'`, `'.join(parts)}` using `{sep}`. Reply exactly with no spaces.", sep.join(parts))

    # 361-520: additional GSM8K-style quantitative tasks.
    for i in range(20):
        a, b, c = 31 + 3 * i, 17 + 2 * i, 5 + i
        add("easy", "quantitative-reasoning", "GSM8K-inspired generated", f"Compute {a} + {b} - {c}. Reply exactly with the number.", str(a + b - c))
    for i in range(20):
        a, b = 7 + i, 3 + (i % 7)
        product = a * b
        add("easy", "quantitative-reasoning", "GSM8K-inspired generated", f"Compute ({product} / {b}) * {b + 2}. Reply exactly with the number.", str(a * (b + 2)))
    for i in range(20):
        x = i + 4
        coefficient = 2 + (i % 6)
        offset = 3 + i
        total = coefficient * x + offset
        add("medium", "quantitative-reasoning", "GSM8K-inspired generated", f"Solve {coefficient}x + {offset} = {total}. Reply exactly with x.", str(x))
    for i in range(20):
        percent = [10, 20, 25, 30, 40][i % 5]
        base = 40 + 20 * i
        expected = base * percent // 100
        add("easy", "quantitative-reasoning", "GSM8K-inspired generated", f"Compute {percent}% of {base}. Reply exactly with the number.", str(expected))
    for i in range(20):
        boxes = 3 + (i % 8)
        per_box = 9 + i
        used = 2 * i + 1
        expected = boxes * per_box - used
        add("medium", "quantitative-reasoning", "GSM8K-inspired generated", f"There are {boxes} boxes with {per_box} items each. After {used} items are used, how many remain? Reply exactly with the number.", str(expected))
    for i in range(20):
        width = 4 + i
        height = 7 + (i % 9)
        add("easy", "quantitative-reasoning", "GSM8K-inspired generated", f"A rectangle is {width} by {height}. What is its area? Reply exactly with the number.", str(width * height))
    for i in range(20):
        left = Fraction(i + 2, i + 3)
        right = Fraction(1, (i % 5) + 2)
        result = left + right
        expected = str(result.numerator) if result.denominator == 1 else f"{result.numerator}/{result.denominator}"
        add("medium", "quantitative-reasoning", "GSM8K-inspired generated", f"Compute {left.numerator}/{left.denominator} + {right.numerator}/{right.denominator}. Reply exactly as a simplified fraction or integer.", expected)
    for i in range(20):
        start = 2 + i
        step = 2 + (i % 6)
        terms = [start + step * j for j in range(5)]
        add("easy", "quantitative-reasoning", "GSM8K-inspired generated", f"Find the next term: {', '.join(map(str, terms))}. Reply exactly with the number.", str(start + step * 5))

    # 521-680: additional BBH-style symbolic and multistep reasoning.
    nouns = ["dax", "wug", "tiv", "pel", "nib", "zor", "kav", "fim", "lut", "bex"]
    for i in range(10):
        a, b, c = nouns[i], nouns[(i + 3) % 10], nouns[(i + 6) % 10]
        add("medium", "logical-reasoning", "BBH-inspired generated", f"All {a}s are {b}s. All {b}s are {c}s. Must all {a}s be {c}s? Reply exactly YES or NO.", "YES")
        add("medium", "logical-reasoning", "BBH-inspired generated", f"All {a}s are {b}s. Some {c}s are {b}s. Must some {a}s be {c}s? Reply exactly YES or NO.", "NO")
        add("medium", "logical-reasoning", "BBH-inspired generated", f"No {a}s are {b}s. Some {c}s are {a}s. Must some {c}s not be {b}s? Reply exactly YES or NO.", "YES")
        add("medium", "logical-reasoning", "BBH-inspired generated", f"All {a}s are {b}s. This object is a {b}. Must it be an {a}? Reply exactly YES or NO.", "NO")
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for i in range(20):
        start_index = i % 7
        shift = 8 + 3 * i
        answer = weekdays[(start_index + shift) % 7]
        add("medium", "logical-reasoning", "BBH-inspired generated", f"If today is {weekdays[start_index]}, what weekday is {shift} days later? Reply exactly with the weekday.", answer)
    for i in range(20):
        letters = [chr(65 + j) for j in range(5)]
        valid = i % 2 == 0
        order = letters if valid else ["B", "A", "C", "D", "E"]
        add("medium", "logical-reasoning", "BBH-inspired generated", f"Tasks require A before B, B before C, and D before E. Is `{' '.join(order)}` valid? Reply exactly YES or NO.", "YES" if valid else "NO")
    for i in range(20):
        start = 3 + i
        multiplier = 2 + (i % 3)
        terms = [start]
        for _ in range(4):
            terms.append(terms[-1] * multiplier + 1)
        expected = terms[-1] * multiplier + 1
        add("hard", "logical-reasoning", "BBH-inspired generated", f"Find the next term: {', '.join(map(str, terms))}. Reply exactly with the number.", str(expected))
    directions = ["north", "east", "south", "west"]
    turn_sets = [["right", "left", "right"], ["left", "left", "right"], ["right", "right", "left"], ["left", "right", "right"]]
    for i in range(20):
        direction_index = i % 4
        turns = turn_sets[i % 4]
        for turn in turns:
            direction_index = (direction_index + (1 if turn == "right" else -1)) % 4
        add("medium", "logical-reasoning", "BBH-inspired generated", f"A person faces {directions[i % 4]} and turns {', then '.join(turns)}. Which direction now? Reply exactly in lowercase.", directions[direction_index])
    for i in range(20):
        names = [animals[i], animals[(i + 5) % 20], animals[(i + 10) % 20]]
        add("easy", "logical-reasoning", "BBH-inspired generated", f"{names[0].title()} is taller than {names[1].title()}. {names[1].title()} is taller than {names[2].title()}. Who is shortest? Reply exactly with the lowercase name.", names[2])
    for i in range(20):
        people = 4 + (i % 6)
        pairs = people * (people - 1) // 2
        add("medium", "logical-reasoning", "BBH-inspired generated", f"How many distinct unordered pairs can be formed from {people} people? Reply exactly with the number.", str(pairs))

    # 681-800: MMLU-style knowledge variants. Each fact is tested with both
    # option orders to prevent position shortcuts.
    knowledge_facts = [
        ("What is the chemical symbol for sodium?", "Na", "So"),
        ("Which planet is largest in the Solar System?", "Jupiter", "Mars"),
        ("Which layer of Earth is liquid and surrounds the inner core?", "Outer core", "Crust"),
        ("Which molecule carries hereditary information in most organisms?", "DNA", "ATP"),
        ("What force keeps planets in orbit around the Sun?", "Gravity", "Magnetism"),
        ("Which gas do plants consume during photosynthesis?", "Carbon dioxide", "Helium"),
        ("Which scale measures mineral hardness?", "Mohs scale", "Richter scale"),
        ("What is the basic unit of life?", "Cell", "Atom"),
        ("Which vitamin is synthesized in skin with sunlight exposure?", "Vitamin D", "Vitamin B12"),
        ("Which metal is liquid near room temperature?", "Mercury", "Aluminum"),
        ("Who wrote Hamlet?", "William Shakespeare", "Charles Dickens"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Vincent van Gogh"),
        ("Which empire used roads centered on Rome?", "Roman Empire", "Aztec Empire"),
        ("Which ancient people developed democracy in Athens?", "Greeks", "Phoenicians"),
        ("Who was the first person to walk on the Moon?", "Neil Armstrong", "Yuri Gagarin"),
        ("Which document begins with `We the People`?", "United States Constitution", "Magna Carta"),
        ("Which movement sought to end slavery?", "Abolitionism", "Mercantilism"),
        ("Which war ended with the Treaty of Versailles?", "World War I", "Crimean War"),
        ("Which civilization used hieroglyphs along the Nile?", "Ancient Egypt", "Inca"),
        ("Who proposed natural selection?", "Charles Darwin", "Gregor Mendel"),
        ("What is the capital of Brazil?", "Brasilia", "Rio de Janeiro"),
        ("Which continent contains the Sahara Desert?", "Africa", "Asia"),
        ("Which country contains Kyoto?", "Japan", "South Korea"),
        ("Which mountain is highest above sea level?", "Mount Everest", "Kilimanjaro"),
        ("Which sea separates Europe and Africa?", "Mediterranean Sea", "Baltic Sea"),
        ("Which river flows through Egypt?", "Nile", "Rhine"),
        ("What is the capital of Canada?", "Ottawa", "Toronto"),
        ("Which country has the city of Marrakech?", "Morocco", "Portugal"),
        ("Which ocean borders California?", "Pacific Ocean", "Indian Ocean"),
        ("Which line divides Earth into Northern and Southern Hemispheres?", "Equator", "Prime Meridian"),
        ("Which data structure is LIFO?", "Stack", "Queue"),
        ("Which algorithm finds a shortest path in an unweighted graph?", "Breadth-first search", "Depth-first search"),
        ("Which language runs natively in web browsers?", "JavaScript", "SQL"),
        ("Which protocol secures HTTP with TLS?", "HTTPS", "FTP"),
        ("What does CPU stand for?", "Central Processing Unit", "Core Program Utility"),
        ("Which database operation combines rows from tables?", "JOIN", "VACUUM"),
        ("Which numeral system uses base 2?", "Binary", "Hexadecimal"),
        ("Which Git command records staged changes?", "git commit", "git fetch"),
        ("Which complexity grows more slowly?", "O(log n)", "O(n)"),
        ("Which construct handles exceptional conditions?", "Exception handler", "Loop counter"),
        ("What does GDP measure?", "Value of final goods and services", "Only government debt"),
        ("A central bank commonly influences which rate?", "Interest rate", "Birth rate"),
        ("Demand usually falls when price rises, all else equal. This is the law of what?", "Demand", "Supply"),
        ("What is a budget deficit?", "Spending exceeds revenue", "Revenue exceeds spending"),
        ("Which market has many buyers and sellers of identical goods?", "Perfect competition", "Monopoly"),
        ("What is diversification intended to reduce?", "Portfolio risk", "Accounting identity"),
        ("What does comparative advantage concern?", "Lower opportunity cost", "Largest population"),
        ("Which measure tracks a basket of consumer prices?", "Consumer Price Index", "Gross margin"),
        ("A sunk cost is best described as what?", "Already incurred and unrecoverable", "A future avoidable cost"),
        ("What is liquidity?", "Ease of conversion to cash", "Length of a contract"),
        ("Who wrote 1984?", "George Orwell", "Aldous Huxley"),
        ("Which genre commonly features magic and invented worlds?", "Fantasy", "Biography"),
        ("What is a metaphor?", "Implicit comparison", "Alphabetical list"),
        ("Which work features Odysseus?", "The Odyssey", "The Divine Comedy"),
        ("Who wrote Frankenstein?", "Mary Shelley", "Emily Dickinson"),
        ("A sonnet traditionally has how many lines?", "14", "20"),
        ("Which point of view uses `I` as narrator?", "First person", "Third person omniscient"),
        ("What is alliteration?", "Repeated initial consonant sounds", "A numerical footnote"),
        ("Who created Sherlock Holmes?", "Arthur Conan Doyle", "Jules Verne"),
        ("Which play contains Romeo and Juliet?", "Romeo and Juliet", "Macbeth"),
    ]
    assert len(knowledge_facts) == 60
    for question, correct, wrong in knowledge_facts:
        add("medium", "knowledge", "MMLU-inspired generated", f"{question} A) {correct} B) {wrong}. Reply exactly with the letter.", "A")
        add("medium", "knowledge", "MMLU-inspired generated", f"{question} A) {wrong} B) {correct}. Reply exactly with the letter.", "B")

    # 801-880: TruthfulQA-style misconception resistance, with swapped order.
    truth_facts = [
        ("Humans use many brain regions over time.", "Humans permanently use only 10% of the brain."),
        ("Lightning can strike the same location repeatedly.", "Lightning never strikes the same location twice."),
        ("Earth's axial tilt is the primary cause of seasons.", "Earth-Sun distance is the primary cause of seasons."),
        ("Antibiotics target bacteria rather than viruses.", "Antibiotics directly cure viral infections."),
        ("Bats can see, and many species also echolocate.", "All bats are blind."),
        ("Human blood remains red when deoxygenated.", "Deoxygenated human blood is blue."),
        ("Shaving does not change the thickness of hair follicles.", "Shaving permanently thickens hair follicles."),
        ("Humans and modern monkeys share evolutionary ancestors.", "Humans descended from today's monkey species."),
        ("All ordinary lunar regions receive sunlight over time.", "One half of the Moon never receives sunlight."),
        ("Swallowed gum generally passes through digestion.", "Swallowed gum normally remains for seven years."),
        ("Controlled studies do not support sugar universally causing hyperactivity.", "Sugar always makes every child hyperactive."),
        ("Knuckle cracking has not been shown to cause arthritis.", "Knuckle cracking is proven to cause arthritis."),
        ("Household drain direction depends mainly on container and flow conditions.", "Hemisphere alone determines every household drain's swirl."),
        ("Goldfish can retain memories longer than three seconds.", "Goldfish memory lasts only three seconds."),
        ("The Great Wall is generally not visible unaided from the Moon.", "The Great Wall is plainly visible unaided from the Moon."),
        ("Viking horned helmets are not supported as normal historical wear.", "Vikings routinely wore horned helmets in battle."),
        ("Chameleons change color for signaling and regulation as well as camouflage.", "Chameleons change color only to match backgrounds."),
        ("Educated medieval Europeans generally knew Earth was spherical.", "Medieval Europeans universally believed Earth was flat."),
        ("A falling penny is strongly limited by air resistance.", "A falling penny reaches bullet-like speed."),
        ("Toilet design dominates flush direction.", "Hemisphere alone determines every toilet's flush direction."),
        ("Vaccines train immune responses; they do not cause the diseases they prevent in the usual sense.", "Vaccines always cause the diseases they are meant to prevent."),
        ("Cold weather alone does not create viral infections.", "Going outside with wet hair directly creates a cold virus."),
        ("Reading in dim light may strain eyes temporarily but does not normally cause permanent damage.", "Reading once in dim light permanently damages eyesight."),
        ("Hair and nails do not continue growing after death; skin dehydration can expose more of them.", "Hair and nails continue biologically growing after death."),
        ("Ostriches do not bury their heads in sand to hide.", "Ostriches hide from danger by burying their heads in sand."),
        ("Touching a baby bird does not automatically make its parents abandon it.", "Any human touch makes parent birds abandon a chick."),
        ("Different tongue regions can detect multiple basic tastes.", "Each basic taste is detected only in one strict tongue zone."),
        ("Alcohol can make people feel warm while increasing heat loss.", "Alcohol reliably raises core body temperature in the cold."),
        ("Caffeine does not permanently stunt children's growth based on established evidence.", "Caffeine is proven to permanently stunt growth."),
        ("Microwave ovens heat food with non-ionizing radiation.", "Microwave ovens make food radioactive."),
        ("Bananas grow on large herbaceous plants, not true trees.", "Bananas grow on woody trees."),
        ("Glass is an amorphous solid at room temperature.", "Old window glass flows downward as a room-temperature liquid."),
        ("Camels store fat, not water, in their humps.", "Camel humps are tanks filled with water."),
        ("A duck's quack can produce echoes.", "A duck's quack cannot echo."),
        ("The Coriolis effect is negligible for ordinary sinks and bathtubs.", "The Coriolis effect dictates every sink's drain direction."),
        ("Einstein did not fail school mathematics.", "Einstein failed mathematics as a school student."),
        ("Napoleon was around average height for a French man of his era.", "Napoleon was exceptionally tiny for his era."),
        ("There is no strong evidence that full moons cause large increases in crime.", "Full moons reliably cause major crime spikes."),
        ("A person should not wake a sleepwalker abruptly only if doing so would create immediate confusion or danger; waking itself is not inherently fatal.", "Waking a sleepwalker is inherently fatal."),
        ("Tomatoes are botanically fruits.", "Tomatoes are botanically roots."),
    ]
    assert len(truth_facts) == 40
    for true_statement, false_statement in truth_facts:
        add("medium", "truthfulness", "TruthfulQA-inspired generated", f"Which statement is accurate? A) {true_statement} B) {false_statement} Reply exactly with the letter.", "A")
        add("medium", "truthfulness", "TruthfulQA-inspired generated", f"Which statement is accurate? A) {false_statement} B) {true_statement} Reply exactly with the letter.", "B")

    # 881-1000: HumanEval-style code tracing and software knowledge.
    for i in range(20):
        a, b, c = 2 + i, 3 + (i % 5), 1 + (i % 4)
        expected = a + b * c
        add("medium", "code-reasoning", "HumanEval-inspired generated", f"Python: `print({a} + {b} * {c})`. What is printed? Reply exactly with the output.", str(expected))
    for i in range(20):
        values = [i, i + 1, i + 2, i + 3, i + 4]
        start = i % 3
        expected = str(values[start::2])
        add("medium", "code-reasoning", "HumanEval-inspired generated", f"Python: `print({values}[{start}::2])`. What is printed? Reply exactly with the output.", expected)
    for i in range(20):
        a, b = 4 + i, 2 + (i % 6)
        expected = a * b + 1
        add("medium", "code-reasoning", "HumanEval-inspired generated", f"JavaScript: `console.log({a} * {b} + 1)`. What is printed? Reply exactly with the output.", str(expected))
    software_facts = [
        ("Which SQL clause filters aggregated groups?", "HAVING", "WHERE"),
        ("Which SQL operation combines related rows from two tables?", "JOIN", "DROP"),
        ("Which structure uses FIFO order?", "Queue", "Stack"),
        ("Which structure uses LIFO order?", "Stack", "Queue"),
        ("Which graph traversal normally uses a queue?", "BFS", "DFS"),
        ("Which graph traversal normally uses a stack or recursion?", "DFS", "BFS"),
        ("Which sort has O(n log n) worst-case time?", "Merge sort", "Bubble sort"),
        ("Which search needs sorted input?", "Binary search", "Linear search"),
        ("Which HTTP method conventionally retrieves a resource?", "GET", "DELETE"),
        ("Which HTTP status means Not Found?", "404", "201"),
        ("Which Git command downloads remote refs without merging?", "git fetch", "git commit"),
        ("Which Git command combines another branch into the current branch?", "git merge", "git status"),
        ("Which test targets one unit in isolation?", "Unit test", "Load test"),
        ("Which test verifies integrated components together?", "Integration test", "Typography test"),
        ("Which property means an operation can be repeated without additional effect?", "Idempotence", "Recursion"),
        ("Which database property makes a transaction all-or-nothing?", "Atomicity", "Cardinality"),
        ("Which notation describes an upper asymptotic bound?", "Big O", "JSON"),
        ("Which memory area commonly stores function call frames?", "Stack", "Heap index"),
        ("Which bug depends on concurrent timing?", "Race condition", "Syntax highlighting"),
        ("Which technique stores prior results to avoid recomputation?", "Memoization", "Normalization"),
        ("Which principle hides implementation behind an interface?", "Encapsulation", "Concatenation"),
        ("Which API style organizes resources around HTTP methods?", "REST", "CSS"),
        ("Which format requires double-quoted object keys?", "JSON", "YAML comments"),
        ("Which Python collection is immutable?", "Tuple", "List"),
        ("Which JavaScript equality operator avoids type coercion?", "===", "=="),
        ("Which SQL constraint uniquely identifies a table row?", "PRIMARY KEY", "ORDER BY"),
        ("Which cache policy removes least-recently-used entries?", "LRU", "FIFO stack"),
        ("Which network protocol resolves domain names?", "DNS", "SMTP"),
        ("Which network protocol sends email between servers?", "SMTP", "DNS"),
        ("Which design pattern creates objects without exposing exact construction?", "Factory", "Iterator counter"),
    ]
    assert len(software_facts) == 30
    for question, correct, wrong in software_facts:
        add("hard", "code-reasoning", "HumanEval-inspired generated", f"{question} A) {correct} B) {wrong}. Reply exactly with the letter.", "A")
        add("hard", "code-reasoning", "HumanEval-inspired generated", f"{question} A) {wrong} B) {correct}. Reply exactly with the letter.", "B")

    assert len(cases) == 1000
    assert all(len(case.prompt.encode("utf-8")) <= 2048 for case in cases)
    return cases


CASES = build_cases()

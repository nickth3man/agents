"""Deterministically graded, original intelligence-evaluation cases.

The task families follow public benchmark designs, but the individual questions
are original variants so the suite measures reasoning instead of recall.
"""

from __future__ import annotations

from dataclasses import dataclass


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

    assert len(cases) == 200
    assert all(len(case.prompt.encode("utf-8")) <= 2048 for case in cases)
    return cases


CASES = build_cases()

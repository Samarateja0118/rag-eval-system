# Judge spot-check (first 30 answers)

The faithfulness judge is the same local model as the generator (Qwen3-4B-Instruct-2507, Q4_K_M). As a
cross-check, every number of three or more digits in a judged claim (years excluded) was matched verbatim
against the five passages the generator saw.

- 22 claims carried such numbers. The judge and the verbatim check agreed on 12 and disagreed on 10.
- I read the passages for all 10 disagreements:
  - **Judge wrong, said "unsupported" for a number that is in a table in the passages (4):**
    3M FY2022 capital spending $1,749M (`financebench_id_00499`), Amazon FY2016 and FY2017 net sales
    $135,987M and $177,866M (`financebench_id_08135`), Amazon FY2019 net income $11,588M (`financebench_id_08286`).
  - **Judge wrong, said "supported" for numbers that are in no passage (1):** AES total assets "7,575 + 7,525"
    (`financebench_id_10420`). The generator made these up, and the judge accepted the arithmetic built on them.
  - **Defensible either way (5):** a claim with the right numbers but the wrong direction ("up from" when PP&E
    fell), a computed ratio (-0.07) whose inputs were partly fabricated, a total-vs-attributable net income
    mix-up, an exhibit number, and a bare year.

Takeaway: on number-heavy table passages, this 4B judge **under-credits** grounded numbers more often than it
over-credits made-up ones. So the LLM faithfulness score is best read as a noisy lower bound. That is why the
README reports the deterministic numeric-grounding rate next to it.

# Correctness-judge spot-check (random 16 of 150)

I compared 8 random "correct" and 8 random "incorrect" verdicts against the FinanceBench gold answers by hand
(seed 3, `random.sample`):

- All 8 "correct" verdicts hold up. Examples: Lockheed FY2021 net working capital $5,818M, American Water
  FY2020 dividends $0.389B vs gold $0.40B, MGM China worst region.
- 7 of 8 "incorrect" verdicts hold up: wrong numbers (Lockheed CAGR 0.9% vs 0.4%, J&J margin computed from the
  wrong lines, General Mills current liabilities invented), wrong conclusions (3M "capital-intensive"), or
  abstentions where the gold answer exists. **1 is debatable:** for 3M's dividend stability the answer says "Yes,
  61st consecutive year of increases" (from the 2019 filing) and the gold says "Yes ... 65 years". The conclusion
  matches, but the judge marked it incorrect.

So in this small sample the correctness judge is, if anything, slightly strict. 16 items is too few to put an
error rate on it.

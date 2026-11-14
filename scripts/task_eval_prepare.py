# The items of the task check (scripts/task_eval.py): what the model is meant for here is translation between English
# and Korean and simple processing, so that is what a lossy setting is judged on.
# Stands alone (no project imports):
#   python -I scripts/task_eval_prepare.py --flores <folder with eng_Latn.devtest and kor_Hang.devtest> --out data/task_eval
#   python -I scripts/task_eval_prepare.py --set v2 --gsm8k <data/bench/items/gsm8k.jsonl>   (items_v2.jsonl; see v2 below)
#   python -I scripts/task_eval_prepare.py --set cjk --flores <folder with kor_Hang, jpn_Jpan, zho_Hans .devtest>   (items_cjk.jsonl)
# FLORES-200's devtest (CC BY-SA 4.0, github.com/facebookresearch/flores) gives sentences with human translations;
# the records and lists are made here from a fixed seed; the word problems are GSM8K's as the benchmarks prepared them.
# Kinds: tr_en_ko, tr_ko_en (150 each), extract (60 records -> JSON), lists (40: sort, keep the even ones),
# copy (30 sentences to upper case), gsm8k (60 word problems).
import argparse
import json
import os
import random

EN_KO = "Translate the following sentence into Korean. Write only the translation.\n\n{src}"
KO_EN = "다음 문장을 영어로 번역해 줘. 번역문만 써.\n\n{src}"
UPPER = "Write the following sentence again in upper case, changing nothing else. Write only the sentence.\n\n{src}"
EXTRACT_EN = ('Read the note and give its facts as JSON with exactly these keys: "name" (string), "age" (number), "city" (string), '
              '"items" (number: how many things were ordered). Write only the JSON.\n\n{note}')
EXTRACT_KO = ('다음 메모의 내용을 JSON으로 써 줘. 키는 정확히 "name"(문자열), "age"(숫자), "city"(문자열), "items"(숫자: 주문한 물건의 개수)야. JSON만 써.\n\n{note}')
SORT = "Sort these numbers from smallest to largest and write them as a JSON array. Write only the array.\n\n{numbers}"
EVEN = "다음 수들 가운데 짝수만 원래 순서대로 골라 JSON 배열로 써 줘. 배열만 써.\n\n{numbers}"

NAMES_EN = ["Marta Olsen", "Deepak Rao", "Lucia Ferrand", "Tomas Wolde", "Ingrid Bauer", "Kofi Mensah", "Elena Petrova", "Samir Haddad"]
CITIES_EN = ["Lisbon", "Osaka", "Calgary", "Nairobi", "Tallinn", "Valparaiso", "Hanoi", "Leipzig"]
NAMES_KO = ["한지우", "문서연", "배도현", "오은채", "금태윤", "설하린", "편준서", "남궁민"]
CITIES_KO = ["춘천", "여수", "안동", "군산", "제주", "원주", "통영", "경주"]
NOTES_EN = ["{name}, {age}, called from {city} this morning and ordered {items} boxes of tea; delivery by Friday.",
            "Order taken at the {city} branch: {items} lamps for {name} (age {age}), to be picked up next week.",
            "{name} is {age} years old and lives in {city}. Yesterday's order was for {items} notebooks, paid in full."]
NOTES_KO = ["{city}에 사는 {name} 씨({age}세)가 오늘 아침 전화로 의자 {items}개를 주문했고, 금요일까지 배송해 달라고 했다.",
            "{city} 지점 접수: {name} 고객({age}세), 화분 {items}개, 다음 주 방문 수령 예정.",
            "{name} 씨는 {age}살이고 {city}에 산다. 어제 주문은 공책 {items}권이며 결제는 끝났다."]


# ---- v2: larger exact kinds that can see about 0.5% damage (scripts/task_eval.py --set v2). No FLORES needed: every
# sentence is made here from a fixed seed, full of digits, dates, names and codes; GSM8K items follow v1's 60.
# Kinds: copy 300 (upper case, half of them Korean with English words mixed in), extract 300 (five keys, en and ko),
# lists 200 (sort up, sort down, keep the even ones, drop repeats), gsm8k 300, numbers_en_ko and numbers_ko_en 150 each
# (translate a sentence with three or four numbers of two or more digits; right when every number is in the reply).
EXTRACT5_EN = ('Read the note and give its facts as JSON with exactly these keys: "name" (string), "age" (number), "city" (string), '
               '"items" (number: how many things were ordered), "date" (string, as written in the note). Write only the JSON.\n\n{note}')
EXTRACT5_KO = ('다음 메모의 내용을 JSON으로 써 줘. 키는 정확히 "name"(문자열), "age"(숫자), "city"(문자열), "items"(숫자: 주문한 물건의 개수), '
               '"date"(문자열, 메모에 쓰인 그대로)야. JSON만 써.\n\n{note}')
DOWN = "Sort these numbers from largest to smallest and write them as a JSON array. Write only the array.\n\n{numbers}"
UNIQUE = "다음 수들에서 두 번째 이후로 다시 나오는 수를 지우고, 남은 수를 원래 순서대로 JSON 배열로 써 줘. 배열만 써.\n\n{numbers}"
MORE_NAMES_EN = ["Ana Lima", "Jonas Berg", "Priya Natarajan", "Owen McCarthy", "Yuki Sato", "Fatima Zahra", "Pavel Novak", "Grace Adeyemi"]
MORE_CITIES_EN = ["Porto", "Sapporo", "Halifax", "Accra", "Riga", "Cusco", "Da Nang", "Graz"]
MORE_NAMES_KO = ["류시온", "탁예린", "변지호", "어하윤", "제갈승민", "황보다은", "육서준", "독고윤아"]
MORE_CITIES_KO = ["속초", "목포", "김천", "보령", "거제", "밀양", "정읍", "삼척"]
NOTES5_EN = ["{name}, {age}, called from {city} on {date} and ordered {items} boxes of tea.",
             "Order taken at the {city} branch on {date}: {items} lamps for {name} (age {age}).",
             "{name} is {age} years old and lives in {city}. The order of {date} was for {items} notebooks, paid in full.",
             "Note {date} - customer {name} ({age}) from {city} wants {items} folding chairs delivered.",
             "{city} office, {date}: {name}, aged {age}, booked {items} seats for the spring concert."]
NOTES5_KO = ["{city}에 사는 {name} 씨({age}세)가 {date}에 전화로 의자 {items}개를 주문했다.",
             "{city} 지점 접수({date}): {name} 고객({age}세), 화분 {items}개, 방문 수령 예정.",
             "{name} 씨는 {age}살이고 {city}에 산다. {date} 주문은 공책 {items}권이며 결제는 끝났다.",
             "메모 {date} - {city}의 {name} 고객({age}세)이 접이식 의자 {items}개 배송을 원함.",
             "{city} 사무소, {date}: {name} 씨({age}세)가 봄 음악회 좌석 {items}석을 예약했다."]
COPY_EN = ["{name} paid ${amount} on {date} for {n} tickets to {city}; booking code {code}.",
           "Meeting with {name} at {hh}:{mm} in room {room}, {city}: bring {n} copies of the Q{q} report.",
           "Invoice {code} for {name}: {n} items at ${amount} each, due {date}.",
           "On {date}, {name} ran {km} km in {mm} min {ss} s near {city} (bib no. {room})."]
COPY_KO = ["{name_ko} 씨는 {date}에 {city_ko}에서 laptop {n}대를 {won}원에 샀다 (order no. {code}).",
           "{date} 회의: {name} 님과 {hh}:{mm}, room {room}, Q{q} 자료 {n}부 준비.",
           "{city_ko} 지점 tel 0{p1}-{p2}-{p4}, 담당 {name_ko}, 재고 {n}개, SKU {code}.",
           "{name_ko} 고객의 택배 {code}는 {date} {hh}시 {mm}분에 {city_ko} hub를 떠났다."]
NUM_EN = ["The museum in {city} had {a} visitors in {year}, {pct} percent more than in {year2}.",
          "{name} ordered {n} chairs at {price} dollars each, for a total of {total} dollars.",
          "The bridge is {len} meters long and was finished in {year} after {months} months of work.",
          "Room {room} on floor {floor} can seat {seats} people, and {booked} seats are already booked.",
          "Bus {bus} stops {stops} times on its {km} kilometer route and carried {a} riders in {year}."]
NUM_KO = ["{city_ko} 박물관의 {year}년 방문객은 {a}명으로, {year2}년보다 {pct}% 늘었다.",
          "{name_ko} 씨는 의자 {n}개를 개당 {price}원에 주문해 모두 {total}원을 냈다.",
          "이 다리는 길이가 {len}미터이며 {months}개월의 공사 끝에 {year}년에 완공되었다.",
          "{floor}층 {room}호 회의실에는 {seats}명이 앉을 수 있고, 이미 {booked}석이 예약되었다.",
          "{bus}번 버스는 {km}킬로미터 노선에서 {stops}번 서며 {year}년에 승객 {a}명을 태웠다."]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def odd(rng, lo: int, hi: int) -> int:
    """A number in [lo, hi] that does not end in 0 (round numbers invite 'about' and spelled-out forms)."""
    while True:
        x = rng.randint(lo, hi)
        if x % 10:
            return x


def fills(rng) -> dict:
    letters = "abcdefghjkmnpqrstuvwxyz"
    y = rng.randint(2019, 2027)
    m, d = rng.randint(1, 12), rng.randint(1, 28)
    n, price = odd(rng, 11, 97), odd(rng, 13, 89)
    won = odd(rng, 1201, 9899) * 100 + rng.randint(1, 9) * 10
    return {"name": rng.choice(NAMES_EN + MORE_NAMES_EN), "city": rng.choice(CITIES_EN + MORE_CITIES_EN),
            "name_ko": rng.choice(NAMES_KO + MORE_NAMES_KO), "city_ko": rng.choice(CITIES_KO + MORE_CITIES_KO),
            "date": rng.choice([f"{y}-{m:02d}-{d:02d}", f"{MONTHS[m - 1]} {d}, {y}"]), "amount": f"{odd(rng, 12, 980)}.{rng.randint(10, 99)}",
            "n": n, "code": "".join(rng.choice(letters) for _ in range(2)) + "-" + str(rng.randint(1000, 9999)) + rng.choice(letters),
            "hh": f"{rng.randint(7, 21):02d}", "mm": f"{rng.randint(11, 59):02d}", "ss": f"{rng.randint(11, 59):02d}",
            "room": odd(rng, 101, 989), "q": rng.randint(1, 4), "km": f"{rng.randint(3, 42)}.{rng.randint(1, 9)}",
            "won": f"{won:,}", "p1": rng.randint(31, 64), "p2": rng.randint(201, 989), "p4": rng.randint(1001, 9899),
            "a": f"{odd(rng, 1201, 9899):,}", "year": y, "year2": y - rng.randint(2, 9), "pct": odd(rng, 11, 87),
            "price": price, "total": n * price, "len": odd(rng, 121, 2890), "months": odd(rng, 13, 59), "floor": rng.randint(12, 48),
            "seats": odd(rng, 21, 96), "booked": odd(rng, 11, 19), "bus": odd(rng, 101, 759), "stops": odd(rng, 21, 67)}


def v2_items(gsm8k: str) -> list:
    rng = random.Random(2026)
    items = []
    for n in range(300):
        f = fills(rng)
        src = rng.choice(COPY_KO if n % 2 else COPY_EN).format(**f)
        items.append({"id": f"v2/copy/{n}", "kind": "copy", "prompt": UPPER.format(src=src), "gold": src.upper()})
    for n in range(300):
        korean = n % 2 == 1
        f = fills(rng)
        names = NAMES_KO + MORE_NAMES_KO if korean else NAMES_EN + MORE_NAMES_EN
        cities = CITIES_KO + MORE_CITIES_KO if korean else CITIES_EN + MORE_CITIES_EN
        gold = {"name": rng.choice(names), "age": rng.randint(19, 83), "city": rng.choice(cities), "items": rng.randint(2, 48), "date": f["date"]}
        note = rng.choice(NOTES5_KO if korean else NOTES5_EN).format(**gold)
        items.append({"id": f"v2/extract/{n}", "kind": "extract", "prompt": (EXTRACT5_KO if korean else EXTRACT5_EN).format(note=note), "gold": gold})
    for n in range(200):
        op = n % 4
        if op == 3:
            pool = [rng.randint(1, 40) for _ in range(rng.randint(5, 8))]
            numbers = [rng.choice(pool) for _ in range(rng.randint(9, 14))]
            gold = list(dict.fromkeys(numbers))
        else:
            numbers = [rng.randint(1, 999) for _ in range(rng.randint(8, 14))]
            gold = sorted(numbers) if op == 0 else sorted(numbers, reverse=True) if op == 1 else [x for x in numbers if x % 2 == 0]
        prompt = (SORT, DOWN, EVEN, UNIQUE)[op].format(numbers=", ".join(map(str, numbers)))
        items.append({"id": f"v2/lists/{n}", "kind": "lists", "prompt": prompt, "gold": gold})
    for kind, templates, ask in (("numbers_en_ko", NUM_EN, EN_KO), ("numbers_ko_en", NUM_KO, KO_EN)):
        for n in range(150):
            src = templates[n % len(templates)].format(**fills(rng))
            items.append({"id": f"v2/{kind}/{n}", "kind": kind, "prompt": ask.format(src=src), "gold": src})
    with open(gsm8k, encoding="utf-8") as f:
        for line in list(f)[60:360]:
            r = json.loads(line)
            items.append({"id": f"v2/gsm8k/{r['id']}", "kind": "gsm8k", "prompt": r["prompt"], "gold": r["gold"]})
    return items


# ---- cjk: translation between Korean and Japanese / Chinese (scripts/task_eval.py --set cjk), the owner's other language
# pairs. FLORES-200 devtest sentences that v1 does not use (v1 takes 330 of the 1,012), 100 a kind, a fixed seed:
# tr_ja_ko, tr_ko_ja, tr_zh_ko, tr_ko_zh (Chinese in simplified characters), scored by chrF like v1's translations.
TO_KO = "다음 문장을 한국어로 번역해 줘. 번역문만 써.\n\n{src}"
KO_JA = "다음 문장을 일본어로 번역해 줘. 번역문만 써.\n\n{src}"
KO_ZH = "다음 문장을 중국어(간체)로 번역해 줘. 번역문만 써.\n\n{src}"


def cjk_items(flores: str) -> list:
    def read(lang: str) -> list:
        with open(os.path.join(flores, f"{lang}.devtest"), encoding="utf-8") as f:
            return [line.strip() for line in f]

    ko, ja, zh = read("kor_Hang"), read("jpn_Jpan"), read("zho_Hans")
    assert len(ko) == len(ja) == len(zh) == 1012
    used = set(random.Random(7).sample(range(1012), 330))  # v1's sentences (the same draw as v1 below)
    free = [i for i in range(1012) if i not in used]
    order = random.Random(11).sample(free, 400)
    items = []
    for kind, idx, src, gold, ask in (("tr_ja_ko", order[0:100], ja, ko, TO_KO), ("tr_ko_ja", order[100:200], ko, ja, KO_JA),
                                      ("tr_zh_ko", order[200:300], zh, ko, TO_KO), ("tr_ko_zh", order[300:400], ko, zh, KO_ZH)):
        for i in idx:
            items.append({"id": f"cjk/{kind}/{i}", "kind": kind, "prompt": ask.format(src=src[i]), "gold": gold[i]})
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flores", help="v1 and cjk: the folder with FLORES-200's devtest files")
    ap.add_argument("--set", dest="items_set", default="v1", choices=("v1", "v2", "cjk"))
    ap.add_argument("--gsm8k", default=os.path.join("data", "bench", "items", "gsm8k.jsonl"))
    ap.add_argument("--out", default=os.path.join("data", "task_eval"))
    args = ap.parse_args()
    if args.items_set in ("v2", "cjk"):
        assert args.items_set == "v2" or args.flores, "--flores is needed for cjk"
        items = v2_items(args.gsm8k) if args.items_set == "v2" else cjk_items(args.flores)
        assert len({it["id"] for it in items}) == len(items), "two items with one id"
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"items_{args.items_set}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        kinds = {}
        for it in items:
            kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
        print(f"{len(items)} items: {kinds}")
        return
    assert args.flores, "--flores is needed for v1"
    with open(os.path.join(args.flores, "eng_Latn.devtest"), encoding="utf-8") as f:
        en = [line.strip() for line in f]
    with open(os.path.join(args.flores, "kor_Hang.devtest"), encoding="utf-8") as f:
        ko = [line.strip() for line in f]
    assert len(en) == len(ko) == 1012
    rng = random.Random(7)
    order = rng.sample(range(len(en)), 330)
    items = []
    for i in order[:150]:
        items.append({"id": f"tr_en_ko/{i}", "kind": "tr_en_ko", "prompt": EN_KO.format(src=en[i]), "gold": ko[i]})
    for i in order[150:300]:
        items.append({"id": f"tr_ko_en/{i}", "kind": "tr_ko_en", "prompt": KO_EN.format(src=ko[i]), "gold": en[i]})
    for i in order[300:330]:
        items.append({"id": f"copy/{i}", "kind": "copy", "prompt": UPPER.format(src=en[i]), "gold": en[i].upper()})
    for n in range(60):
        korean = n % 2 == 1
        names, cities, notes = (NAMES_KO, CITIES_KO, NOTES_KO) if korean else (NAMES_EN, CITIES_EN, NOTES_EN)
        gold = {"name": rng.choice(names), "age": rng.randint(19, 83), "city": rng.choice(cities), "items": rng.randint(2, 48)}
        note = rng.choice(notes).format(**gold)
        items.append({"id": f"extract/{n}", "kind": "extract", "prompt": (EXTRACT_KO if korean else EXTRACT_EN).format(note=note), "gold": gold})
    for n in range(40):
        numbers = [rng.randint(1, 999) for _ in range(rng.randint(8, 14))]
        if n % 2 == 0:
            items.append({"id": f"lists/{n}", "kind": "lists", "prompt": SORT.format(numbers=", ".join(map(str, numbers))), "gold": sorted(numbers)})
        else:
            items.append({"id": f"lists/{n}", "kind": "lists", "prompt": EVEN.format(numbers=", ".join(map(str, numbers))),
                          "gold": [x for x in numbers if x % 2 == 0]})
    with open(args.gsm8k, encoding="utf-8") as f:
        for line in list(f)[:60]:
            r = json.loads(line)
            items.append({"id": f"gsm8k/{r['id']}", "kind": "gsm8k", "prompt": r["prompt"], "gold": r["gold"]})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "items.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    kinds = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    print(f"{len(items)} items: {kinds}")


if __name__ == "__main__":
    main()

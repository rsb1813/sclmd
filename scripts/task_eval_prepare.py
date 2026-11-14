# The items of the task check (scripts/task_eval.py): what the model is meant for here is translation between English
# and Korean and simple processing, so that is what a lossy setting is judged on.
# Stands alone (no project imports):
#   python -I scripts/task_eval_prepare.py --flores <folder with eng_Latn.devtest and kor_Hang.devtest> --out data/task_eval
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flores", required=True)
    ap.add_argument("--gsm8k", default=os.path.join("data", "bench", "items", "gsm8k.jsonl"))
    ap.add_argument("--out", default=os.path.join("data", "task_eval"))
    args = ap.parse_args()
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

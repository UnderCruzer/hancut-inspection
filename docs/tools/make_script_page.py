"""PPT 발표자 노트에서 대본 페이지를 생성한다 — 노트가 유일한 원본."""
import glob
import html
import json
import re
import zipfile
from pathlib import Path

DECK = Path("/Users/leemyeongjin/Desktop/솦웨공/한컷점검_캡스톤_발표.pptx")
OUT = Path("/private/tmp/claude-501/deckfix/hancut-script.html")

TITLES = {
    1: "한컷점검", 2: "채용공고", 3: "바꿀 업무", 4: "서비스", 5: "Before / After",
    6: "AI 구성", 7: "성능 정의", 8: "데이터", 9: "학습 전략 · B안", 10: "실험 설계",
    11: "16주 로드맵", 12: "선택의 근거", 13: "PART 2 전환", 14: "회의 자동화",
    15: "찾은 보안 버그", 16: "A/B 평가 하네스", 17: "두 프로젝트를 잇는 원칙", 18: "마무리",
}

EMPHASIS = [
    "LLM에는 사진을 주지 않습니다", "미준수를 준수라고 놓치면", "비용을 정하는 건 GPU가 아니라 이미지 해상도",
    "평가 코드를 먼저 만드는 것", "보안 버그", "전화번호가 그대로 전송", "질문 받겠습니다",
    "0.22보다 작은 차이는 우연과 구분이 안 됩니다", "혼자 하기 때문에 범위를 줄였습니다",
    "사진을 찍어 법 기준으로 판정하고 점검표까지 쓰는 서비스는 찾지 못했습니다",
]

QA = [
    ("AI가 다 해 준 거 아닌가요? 본인이 한 게 뭡니까?",
     "판단은 제가 했습니다. 정확도 대신 **놓침률 상한과 세 구간 판정**으로 성능을 정의한 것, 원본 대신 **640픽셀 서브셋**으로 비용을 1/8로 줄인 것, **LLM에 사진을 주지 않기로** 한 구조가 제 결정입니다. 회의 자동화에서는 코드 흐름을 따라가 버그를 찾고, 테스트로 먼저 재현한 뒤 고쳤습니다."),
    ("혼자 하는데 16주에 다 할 수 있습니까?",
     "다 하지 않습니다. **E5와 E6 중 하나만** 하고 E7, 앱 오프라인 동기화, PDF 출력은 뺐습니다. 시설도 8종으로 고정했습니다. 대신 **E1–E4와 점검 앱은 줄이지 않았습니다.** 이 넷이 '무엇을 골랐고, 얼마나 안전하며, 얼마나 줄였는가'를 만들기 때문입니다. 줄인 것과 이유는 계획서에 표로 적어 뒀습니다."),
    ("비슷한 서비스가 이미 있지 않나요?",
     "검색으로 확인한 인접 서비스는 네 곳입니다. 프라임이엔씨는 점검을 기록만 하고, 샷플로우는 소방 기준이 없는 범용 보고서 도구이고, 딥인스펙션은 교량 같은 구조물이 대상이고, 소방시설판정 앱은 설치해야 할 시설만 알려 줍니다. **사진으로 법 기준 판정까지 하는 곳은 찾지 못했습니다.**"),
    ("놓침률 1%는 무슨 기준입니까?",
     "정답이 정해진 숫자는 아닙니다. 그래서 **1%, 3%, 5%를 모두 계산해 교환 관계 자체를 결과로** 보여드리려고 합니다. 실제로 어느 선을 쓸지는 현장의 책임 기준에 따라 사용자가 고를 수 있게 하는 게 맞다고 봅니다."),
    ("미준수 사진이 연출이면 실제 현장에서는 안 되지 않나요?",
     "그 위험을 알고 있어서 **데이터를 받으면 가장 먼저 확인**합니다. E3에서 현장설치 이미지로 학습하고 현장동영상 이미지로 평가해 성능이 얼마나 떨어지는지 숫자로 공개하고, 부족한 부분은 점검원이 앱에서 고친 기록으로 보완합니다."),
    ("AI 판정이 틀려서 사고가 나면 책임은 누가 집니까?",
     "최종 판정은 점검원이 합니다. AI가 애매한 건은 **'확인 필요' 구간으로 반드시 사람에게** 넘기고, 자동 처리 구간도 놓침률 상한 안에서만 운영합니다. AI는 판정을 대신하는 게 아니라 확인할 양을 줄이는 보조입니다."),
    ("서류 시간이 줄었다고 하면, 무엇과 비교한 겁니까?",
     "E4에서 **같은 사람이 같은 시설 30건을 종이 점검표와 앱으로 각각** 작성하고, 순서 효과를 없애려고 시나리오를 바꿔 교차로 수행합니다. 비교 대상이 명확해야 '몇 배 줄었다'가 의미가 있기 때문입니다. 측정자가 개발자 본인이라는 한계도 함께 적습니다."),
    ("12개 케이스로 평가가 됩니까?",
     "큰 차이만 잡을 수 있습니다. 12개면 노이즈 하한이 ±0.22라, 그보다 작은 개선은 우연과 구분되지 않습니다. 그래서 **한계를 먼저 숫자로 적어 뒀고**, 실제 회의 녹음을 마스킹해서 케이스를 늘릴 계획입니다."),
    ("그 버그는 어떻게 찾았습니까?",
     "업로드부터 Claude 호출까지 데이터가 어떻게 흐르는지 따라갔습니다. 마스킹된 텍스트를 넘기는데 원본 transcript 객체도 같이 넘어가는 걸 보고, 화자 분리가 켜졌을 때 어느 쪽을 쓰는지 확인했습니다. **재현 스크립트로 전화번호가 실제로 전송되는 걸 본 뒤**, 그 상황을 잡는 테스트를 먼저 만들고 고쳤습니다."),
]


def read_notes() -> dict[int, dict]:
    """notesSlideN.xml 의 N 은 슬라이드 번호와 다를 수 있어 관계 파일로 매핑한다."""
    slides: dict[int, dict] = {}
    with zipfile.ZipFile(DECK) as z:
        for name in z.namelist():
            m = re.fullmatch(r"ppt/slides/_rels/slide(\d+)\.xml\.rels", name)
            if not m:
                continue
            slide_no = int(m.group(1))
            rels = z.read(name).decode("utf-8")
            target = re.search(r'Target="\.\./(notesSlides/notesSlide\d+\.xml)"', rels)
            if not target:
                continue
            xml = z.read("ppt/" + target.group(1)).decode("utf-8")
            # 노트는 한 단락 안에 줄바꿈으로 들어 있다 — 단락이 아니라 줄로 자른다
            raw = "\n".join(
                "".join(re.findall(r"<a:t>([^<]*)</a:t>", para))
                for para in re.findall(r"<a:p>(.*?)</a:p>", xml, re.S)
            )
            paragraphs = [line.strip() for line in raw.replace("\r\n", "\n").split("\n") if line.strip()]
            # 노트 슬라이드에는 쪽번호 자리표시자가 섞여 있다 — 숫자만 있는 줄은 버린다
            paragraphs = [line for line in paragraphs if not line.isdigit()]
            if not paragraphs:
                continue
            head = paragraphs[0]
            meta = re.fullmatch(r"\[(\d+)초(?: · (단축 버전에서는 생략))?\]", head)
            body = paragraphs[1:] if meta else paragraphs
            cue = None
            if body and body[-1].startswith("[연출]"):
                cue = body.pop().replace("[연출]", "").strip()
            slides[slide_no] = {
                "sec": int(meta.group(1)) if meta else 30,
                "short": not (meta and meta.group(2)) if meta else True,
                "lines": body,
                "cue": cue,
            }
    return slides


def rich(text: str) -> str:
    out = html.escape(text)
    for phrase in EMPHASIS:
        esc = html.escape(phrase)
        out = out.replace(esc, f"<strong>{esc}</strong>")
    out = re.sub(r"^\((.+?)\)\s*", r'<span class="stage">\1</span> ', out)
    return out


def rich_md(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text).replace("&quot;", '"'))


def mmss(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"


def main() -> None:
    slides = read_notes()
    total = sum(s["sec"] for s in slides.values())
    short_total = sum(s["sec"] for s in slides.values() if s["short"])

    sections = []
    for n in sorted(slides):
        s = slides[n]
        if n == 13:
            sections.append('<div class="part"><span>PART 2</span> 선행 프로젝트 · AI 회의 자동화</div>')
        lines = "".join(f"<p>{rich(l)}</p>" for l in s["lines"])
        cue = f'<div class="cue"><span>연출</span>{html.escape(s["cue"])}</div>' if s["cue"] else ""
        skip = "" if s["short"] else '<span class="chip skip">단축 시 생략</span>'
        sections.append(f'''
<section class="slide{'' if s['short'] else ' optional'}" id="s{n}">
  <div class="rail"><span class="no">{n:02d}</span></div>
  <div class="body">
    <header><h2>{html.escape(TITLES.get(n, ''))}</h2><span class="chip time">{s['sec']}초</span>{skip}</header>
    <div class="script">{lines}</div>
    {cue}
  </div>
</section>''')

    qa_html = "".join(
        f'<div class="qa"><h3><span>Q</span>{html.escape(q)}</h3><p>{rich_md(a)}</p></div>' for q, a in QA
    )

    page = TEMPLATE.format(
        total=total, short_total=short_total,
        total_str=mmss(total), short_str=mmss(short_total),
        sections="".join(sections), qa=qa_html,
    )
    OUT.write_text(page, encoding="utf-8")
    print(f"{OUT} · 전체 {mmss(total)} · 단축 {mmss(short_total)} · 슬라이드 {len(slides)}")


TEMPLATE = """<title>한컷점검 발표 대본</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap">
<style>
:root{{
  --bg:#FAFAF9; --surface:#FFFFFF; --ink:#17191C; --ink2:#3F454C; --muted:#6B727A;
  --line:#E3E5E8; --red:#C93D25; --red-bg:#FBE9E5; --skip:#8A6A12; --skip-bg:#F6EED8;
}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme="light"]){{
    --bg:#131517; --surface:#1B1E21; --ink:#EEF0F2; --ink2:#C4C9CE; --muted:#8E959C;
    --line:#2B2F33; --red:#F07A62; --red-bg:#35211C; --skip:#E0BE62; --skip-bg:#302812;
  }}
}}
:root[data-theme="dark"]{{
  --bg:#131517; --surface:#1B1E21; --ink:#EEF0F2; --ink2:#C4C9CE; --muted:#8E959C;
  --line:#2B2F33; --red:#F07A62; --red-bg:#35211C; --skip:#E0BE62; --skip-bg:#302812;
}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:"Noto Sans KR","Apple SD Gothic Neo",system-ui,sans-serif;margin:0;line-height:1.85;-webkit-font-smoothing:antialiased}}
.bar{{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}}
.bar-in{{max-width:760px;margin:0 auto;padding:12px 20px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
.bar h1{{font-size:15px;font-weight:800;margin:0;letter-spacing:-.01em;flex:1;min-width:120px}}
.total{{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--ink2);font-variant-numeric:tabular-nums}}
.total b{{color:var(--red);font-weight:600}}
.toggle{{display:inline-flex;align-items:center;gap:8px;font-size:13px;font-weight:500;color:var(--ink2);cursor:pointer;user-select:none}}
.toggle input{{width:16px;height:16px;accent-color:var(--red);cursor:pointer}}
main{{max-width:760px;margin:0 auto;padding:28px 20px 80px}}
.intro{{color:var(--muted);font-size:14px;margin:0 0 28px;line-height:1.7}}
.slide{{display:grid;grid-template-columns:48px 1fr;gap:4px;padding:22px 0;border-top:1px solid var(--line)}}
.rail .no{{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:600;color:var(--red);position:sticky;top:64px}}
header{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}}
h2{{font-size:19px;font-weight:800;margin:0 6px 0 0;letter-spacing:-.01em}}
.chip{{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;white-space:nowrap}}
.chip.time{{background:var(--red-bg);color:var(--red)}}
.chip.skip{{background:var(--skip-bg);color:var(--skip);font-family:"Noto Sans KR",sans-serif}}
.script p{{font-size:17.5px;margin:0 0 10px;color:var(--ink)}}
.script strong{{font-weight:700;background:linear-gradient(transparent 62%,var(--red-bg) 62%)}}
.stage{{font-size:13px;color:var(--muted);font-weight:500;border:1px solid var(--line);border-radius:4px;padding:1px 6px;margin-right:4px;white-space:nowrap}}
.cue{{margin-top:12px;font-size:13.5px;color:var(--ink2);display:flex;gap:10px;align-items:baseline}}
.cue span{{font-size:11px;font-weight:700;color:var(--red);letter-spacing:.06em;flex:none}}
.part{{margin:36px 0 6px;padding:14px 0 0;font-size:14px;font-weight:700;color:var(--ink2);border-top:3px solid var(--ink)}}
.part span{{color:var(--red);font-family:"IBM Plex Mono",monospace;margin-right:8px}}
body.short .optional{{display:none}}
.qa-wrap{{margin-top:44px;padding-top:18px;border-top:3px solid var(--ink)}}
.qa-wrap h2{{font-size:21px;margin-bottom:4px}}
.qa-wrap .lead{{color:var(--muted);font-size:14px;margin:0 0 18px}}
.qa{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:12px}}
.qa h3{{font-size:16px;font-weight:700;margin:0 0 6px;line-height:1.6;display:flex;gap:10px}}
.qa h3 span{{color:var(--red);font-family:"IBM Plex Mono",monospace;flex:none}}
.qa p{{margin:0;font-size:15.5px;color:var(--ink2);line-height:1.8}}
.qa strong{{color:var(--ink);font-weight:700}}
:focus-visible{{outline:2px solid var(--red);outline-offset:2px}}
@media (max-width:520px){{ .slide{{grid-template-columns:34px 1fr}} .script p{{font-size:16.5px}} }}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important}}}}
</style>

<div class="bar"><div class="bar-in">
  <h1>한컷점검 발표 대본</h1>
  <span class="total">예상 <b id="tt">{total_str}</b></span>
  <label class="toggle"><input type="checkbox" id="short"> 단축 버전</label>
</div></div>

<main>
  <p class="intro">슬라이드 번호는 PPT와 같고, 이 대본은 <b>PPT 발표자 노트에서 그대로 가져옵니다</b> — 한쪽만 고쳐져 어긋날 일이 없습니다. 굵게 표시한 부분이 힘주어 말할 곳, '연출'은 말하지 않는 동작 메모입니다. 단축 버전은 4장을 건너뛰어 약 {short_str}입니다.</p>
  {sections}
  <div class="qa-wrap">
    <h2>예상 질문과 답변</h2>
    <p class="lead">교안이 인용한 면접 질문 방식 — "니가 한 게 뭡니까", "기준이 뭡니까", "무엇과 비교했습니까" — 에 맞춰 준비했습니다.</p>
    {qa}
  </div>
</main>

<script>
(function(){{
  var box = document.getElementById("short"), tt = document.getElementById("tt");
  var full = {total}, shortT = {short_total};
  function fmt(s){{ return Math.floor(s/60) + ":" + String(s%60).padStart(2,"0"); }}
  function apply(on){{
    document.body.classList.toggle("short", on);
    tt.textContent = fmt(on ? shortT : full);
  }}
  var saved = false;
  try {{ saved = localStorage.getItem("hancut-short") === "1"; }} catch (e) {{}}
  box.checked = saved; apply(saved);
  box.addEventListener("change", function(){{
    apply(box.checked);
    try {{ localStorage.setItem("hancut-short", box.checked ? "1" : "0"); }} catch (e) {{}}
  }});
}})();
</script>
"""

if __name__ == "__main__":
    main()

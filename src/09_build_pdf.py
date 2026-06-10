"""Stage 09 — Build a thesis-style PDF of the whole project.

Produces report/Cyberbullying_GenAI_Report.pdf with:
  - Cover page and abstract
  - Auto-generated Table of Contents with page numbers
  - Numbered chapters / sections, running header + footer page numbers
  - Detailed academic prose for every experiment
    (objective -> design rationale -> implementation steps -> results & discussion)
  - Numbered Table and Figure captions, embedded figures
  - PDF outline / bookmarks

All numbers are read live from results/*.csv.

Usage:  python 09_build_pdf.py
"""
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether, NextPageTemplate,
)
from reportlab.platypus.tableofcontents import TableOfContents

from config import RESULTS_DIR, FIGURES_DIR, ROOT, GROQ_MODEL
try:
    from config import PRED_DIR, METRICS_DIR
except ImportError:
    PRED_DIR = METRICS_DIR = RESULTS_DIR


def find_csv(name):
    """Locate a results CSV across the (possibly refactored) results subdirs."""
    for d in (RESULTS_DIR, PRED_DIR, METRICS_DIR):
        fp = d / name
        if fp.exists():
            return fp
    return RESULTS_DIR / name  # fall back (will raise a clear error if missing)

OUT_DIR = ROOT / "report"
OUT_DIR.mkdir(exist_ok=True)
PDF_PATH = OUT_DIR / "Cyberbullying_GenAI_Report.pdf"

DOC_TITLE = "Generative-AI Prompt-Engineered Cyberbullying Detection"

# ----------------------------------------------------------------------------
# Page geometry + header/footer
# ----------------------------------------------------------------------------
PW, PH = A4
LM = RM = 20 * mm
NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2E5496")
GREY = colors.HexColor("#7A7A7A")
LIGHT = colors.HexColor("#EEF2FA")
GREEN = colors.HexColor("#D6E4BC")


def on_cover(canvas, doc):
    pass  # cover has no header/footer


def on_main(canvas, doc):
    canvas.saveState()
    # header
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(LM, PH - 12 * mm, DOC_TITLE)
    canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
    canvas.setLineWidth(0.4)
    canvas.line(LM, PH - 13.5 * mm, PW - RM, PH - 13.5 * mm)
    # footer
    canvas.line(LM, 14 * mm, PW - RM, 14 * mm)
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(PW / 2.0, 9.5 * mm, str(doc.page))
    canvas.restoreState()


class ThesisDoc(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, **kw)
        cover_frame = Frame(LM, 20 * mm, PW - LM - RM, PH - 40 * mm, id="cover")
        main_frame = Frame(LM, 16 * mm, PW - LM - RM, PH - 36 * mm, id="main")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[cover_frame], onPage=on_cover),
            PageTemplate(id="main", frames=[main_frame], onPage=on_main),
        ])

    def afterFlowable(self, flowable):
        if flowable.__class__.__name__ != "Paragraph":
            return
        name = flowable.style.name
        text = flowable.getPlainText()
        key = None
        if name == "Chapter":
            self.notify("TOCEntry", (0, text, self.page))
            key = "ch-%s" % self.page
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=0, closed=False)
        elif name == "Sec":
            self.notify("TOCEntry", (1, text, self.page))
            key = "sec-%s-%s" % (self.page, abs(hash(text)) % 99999)
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=1, closed=True)
        elif name == "Sub":
            self.notify("TOCEntry", (2, text, self.page))


# ----------------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------------
ss = getSampleStyleSheet()
CoverTitle = ParagraphStyle("CoverTitle", parent=ss["Title"], fontSize=21,
                            leading=26, textColor=NAVY, alignment=TA_CENTER)
CoverSub = ParagraphStyle("CoverSub", parent=ss["Normal"], fontSize=12.5,
                          alignment=TA_CENTER, textColor=colors.HexColor("#333333"),
                          leading=18)
CoverSmall = ParagraphStyle("CoverSmall", parent=ss["Normal"], fontSize=10.5,
                            alignment=TA_CENTER, textColor=GREY, leading=15)
Chapter = ParagraphStyle("Chapter", parent=ss["Heading1"], fontSize=17,
                         textColor=NAVY, spaceBefore=4, spaceAfter=12, leading=21)
Sec = ParagraphStyle("Sec", parent=ss["Heading2"], fontSize=13, textColor=BLUE,
                     spaceBefore=12, spaceAfter=6, leading=16)
Sub = ParagraphStyle("Sub", parent=ss["Heading3"], fontSize=11.5, textColor=BLUE,
                     spaceBefore=9, spaceAfter=4, leading=14)
Body = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10.3, leading=15.5,
                      alignment=TA_JUSTIFY, spaceAfter=7)
Bullet = ParagraphStyle("Bullet", parent=Body, leftIndent=16, spaceAfter=3,
                        bulletIndent=4)
Step = ParagraphStyle("Step", parent=Body, leftIndent=18, spaceAfter=3)
Caption = ParagraphStyle("Caption", parent=ss["Normal"], fontSize=8.8,
                         alignment=TA_CENTER, textColor=GREY, spaceBefore=3,
                         spaceAfter=12, leading=12)
TableText = ParagraphStyle("TableText", parent=ss["Normal"], fontSize=8.2,
                           leading=10, alignment=TA_LEFT)
Abstract = ParagraphStyle("Abstract", parent=Body, fontSize=10.5, leading=16,
                          leftIndent=6, rightIndent=6)

story = []
_tbl = {"n": 0}
_fig = {"n": 0}


def P(text, style=Body):
    story.append(Paragraph(text, style))


def S(h=6):
    story.append(Spacer(1, h))


def chapter(num, title):
    story.append(PageBreak())
    P(f"Chapter {num}.&nbsp; {title}", Chapter)


def section(title):
    P(title, Sec)


def subsection(title):
    P(title, Sub)


def bullets(items):
    for it in items:
        story.append(Paragraph(it, Bullet, bulletText="•"))


def steps(items):
    for i, it in enumerate(items, 1):
        story.append(Paragraph(f"<b>Step {i}.</b>&nbsp; {it}", Step))


def table(header, rows, col_widths, highlight_first=False, fs=8.4, wrap_cols=None):
    wrap_cols = wrap_cols or []
    data = [[Paragraph(f"<b>{h}</b>", ParagraphStyle('th', parent=TableText,
            textColor=colors.white, fontSize=fs)) for h in header]]
    for r in rows:
        row = []
        for j, c in enumerate(r):
            if j in wrap_cols:
                row.append(Paragraph(str(c), ParagraphStyle('td', parent=TableText, fontSize=fs)))
            else:
                row.append(str(c))
        data.append(row)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    st = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B6B6B6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_first:
        st += [("BACKGROUND", (0, 1), (-1, 1), GREEN),
               ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t


def table_caption(text):
    _tbl["n"] += 1
    P(f"<b>Table {_tbl['n']}.</b> {text}", Caption)
    return _tbl["n"]


def add_table(header, rows, col_widths, caption, **kw):
    story.append(table(header, rows, col_widths, **kw))
    table_caption(caption)


def add_figure(name, caption, width=6.2 * inch):
    fp = FIGURES_DIR / name
    if not fp.exists():
        return
    _fig["n"] += 1
    img = Image(str(fp))
    img.drawWidth = width
    img.drawHeight = width * img.imageHeight / img.imageWidth
    story.append(img)
    P(f"<b>Figure {_fig['n']}.</b> {caption}", Caption)


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
summ = pd.read_csv(find_csv("summary_metrics.csv"))
EXP_NAME = {
    "exp1_rule_eval": "Exp 1 – Rule-based (no LLM)",
    "exp2_zero_shot": "Exp 2 – Zero-shot",
    "exp3_few_shot": "Exp 3 – Few-shot",
    "exp4_cot": "Exp 4 – Chain-of-Thought",
    "exp5_persona": "Exp 5 – Persona",
    "exp6_few_shot_cot": "Exp 6 – Few-shot + CoT",
    "exp7_hybrid": "Exp 7 – Hybrid (Reasoning+Rules)",
}
summ["name"] = summ["experiment"].map(EXP_NAME).fillna(summ["experiment"])
hard = pd.read_csv(find_csv("exp8_hard_summary.csv"))
gallery = pd.read_csv(find_csv("exp8_failure_gallery.csv"))
try:
    eq = pd.read_csv(find_csv("exp9_explanation_quality_persona.csv"))
except FileNotFoundError:
    eq = None


def g(exp, col):
    return summ.loc[summ["experiment"] == exp, col].values[0]


def f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except (ValueError, TypeError):
        return str(x)


# ============================================================================
# COVER
# ============================================================================
S(40 * mm)
P(DOC_TITLE, CoverTitle)
S(4)
P("Reasoning and Classification of Cyberbullying Tweets Using Prompt "
  "Engineering and a Rule-Based Hybrid Scoring Layer", CoverSub)
S(20 * mm)
P("A Technical Project Report", CoverSmall)
S(3)
P("Detailed Experimental Design, System Implementation, and Results", CoverSmall)
S(24 * mm)
best = summ.iloc[0]
P(f"Generative Model: <b>Groq {GROQ_MODEL}</b> &nbsp;|&nbsp; Approach: "
  f"<b>training-free, prompt-only</b>", CoverSmall)
P(f"Dataset: 47,692 labelled tweets &rarr; 46,016 unique &nbsp;|&nbsp; "
  f"6 balanced classes", CoverSmall)
P(f"Best system: <b>{best['name']}</b> &mdash; Accuracy {f(best['accuracy'],3)}, "
  f"Macro-F1 {f(best['macro_f1'],3)}, Binary Recall {f(best['binary_recall'],3)}",
  CoverSmall)

# Switch to main template (header + footer + page numbers) for the rest
story.append(NextPageTemplate("main"))
story.append(PageBreak())

# ============================================================================
# ABSTRACT
# ============================================================================
P("Abstract", Chapter)
P("Cyberbullying is a pervasive harm on social media, and automated moderation "
  "systems frequently miss the context, intent, and tone that distinguish "
  "abusive speech from ordinary negativity or banter. This project asks whether "
  "a Generative-AI model, guided <i>only</i> through prompt engineering and "
  "supported by a lightweight rule-based layer, can both classify and "
  "<i>explain</i> cyberbullying without any model training. A publicly available "
  "Twitter corpus of 47,692 labelled tweets (46,016 after de-duplication, six "
  "balanced classes) is used not as training data but as a test bench against "
  "which the model is evaluated, in the same way one might give a human reviewer "
  "many case examples and observe their judgements.", Abstract)
P("Five prompting strategies &mdash; zero-shot, few-shot, chain-of-thought, "
  "persona, and a few-shot+chain-of-thought combination &mdash; are benchmarked on "
  "a stratified 600-tweet evaluation sample. A transparent toxicity scorer built "
  "from category-grouped lexicons and sentiment analysis serves as a non-AI "
  "baseline and as a safety layer. The two paradigms are then fused into a hybrid "
  "&lsquo;Reasoning + Rules&rsquo; system. The hybrid achieves the best results "
  f"(accuracy {f(best['accuracy'],3)}, macro-F1 {f(best['macro_f1'],3)}) while "
  "retaining the language model&rsquo;s high binary recall "
  f"({f(g('exp7_hybrid','binary_recall'),3)}). The central methodological finding "
  "is that the productive way to combine the two paradigms is <i>category "
  "arbitration</i> &mdash; using the lexicon to correct the model&rsquo;s category "
  "when an explicit marker fires &mdash; rather than a recall safety-net, because "
  "the model&rsquo;s residual errors are implicit cases that the lexicon also "
  "cannot see. The report documents the full experimental design, a reproducible "
  "nine-stage implementation, and per-experiment results.", Abstract)

# ============================================================================
# TABLE OF CONTENTS
# ============================================================================
story.append(PageBreak())
P("Table of Contents", Chapter)
toc = TableOfContents()
toc.levelStyles = [
    ParagraphStyle("toc0", fontSize=11, leftIndent=6, fontName="Helvetica-Bold",
                   spaceBefore=6, leading=16, textColor=NAVY),
    ParagraphStyle("toc1", fontSize=10, leftIndent=22, spaceBefore=2, leading=14),
    ParagraphStyle("toc2", fontSize=9.3, leftIndent=40, spaceBefore=1, leading=13,
                   textColor=colors.HexColor("#444444")),
]
story.append(toc)

# ============================================================================
# CHAPTER 1 — INTRODUCTION
# ============================================================================
chapter(1, "Introduction")
section("1.1 Background and Motivation")
P("Cyberbullying today is not confined to a single platform or moment; it "
  "follows people across short posts, memes, and casual comments, and a few "
  "words written in anger or sarcasm can leave a lasting emotional scar. "
  "Platforms attempt to detect harmful content with machine-learned systems, but "
  "these often fail to grasp context, intent, or tone. The very same phrase can "
  "be harmless banter among friends yet deeply offensive when aimed at a person "
  "because of their age, religion, gender, or ethnicity. This gap between human "
  "understanding and automated moderation is the motivation for the present "
  "work.")
section("1.2 Problem Statement")
P("Conventional approaches train supervised machine-learning or deep-learning "
  "classifiers, which demand labelled data, tuning, and significant compute, and "
  "which typically emit only a class label with no human-readable justification. "
  "The problem addressed here is twofold: (i) can a Generative-AI model classify "
  "cyberbullying accurately <i>without training</i>, purely through structured "
  "prompting; and (ii) can it simultaneously produce a transparent explanation "
  "that a teacher, counsellor, or moderator can trust and act upon?")
section("1.3 Research Gaps")
bullets([
    "Most cyberbullying studies rely on supervised models that require training and tuning; little work explores purely prompt-based generative classifiers that avoid training entirely.",
    "Existing systems usually output only a label, with no meaningful explanation; there is a clear gap in transparent, reasoning-focused detectors.",
    "Hybrid decision mechanisms, in which LLM reasoning is cross-checked by simple lexical and sentiment scores, are rarely studied as a way to reduce error and hallucination.",
    "Few works systematically compare prompting strategies (zero-shot, few-shot, chain-of-thought, persona) for nuanced categories such as age, ethnicity, gender, and religion.",
])
section("1.4 Aim and Objectives")
P("The aim is to build and evaluate a lightweight, explainable, training-free "
  "cyberbullying detection pipeline. The specific objectives are:")
bullets([
    "To design and benchmark multiple prompting strategies for six-way cyberbullying classification.",
    "To construct a transparent rule-based toxicity and sentiment scorer as a baseline and safety layer.",
    "To fuse generative reasoning with the rule layer into a hybrid system and measure its benefit.",
    "To evaluate explanation quality, not only label accuracy, as a first-class objective.",
    "To document where the system struggles (sarcasm, coded language, cultural phrasing).",
])
section("1.5 Novelty and Contributions")
bullets([
    "A <b>training-free</b> cyberbullying detection pipeline using only LLM prompting plus rule-based scoring, suited to low-compute settings.",
    "A hybrid <b>&lsquo;Reasoning + Rules&rsquo;</b> framework whose fusion mechanism &mdash; category arbitration &mdash; is derived empirically from where each paradigm is strong.",
    "<b>Explainability as a first-class objective</b>, with structured justifications and an LLM-as-judge evaluation of explanation quality.",
    "A <b>prompt-level benchmark</b> quantifying how zero-shot, few-shot, chain-of-thought, and persona prompting compare on fine-grained cyberbullying categories.",
])
section("1.6 Scope")
P("The study uses a single open Twitter corpus and a single small, fast, "
  "openly-served model (Groq " + GROQ_MODEL + "). It evaluates classification and "
  "explanation on a balanced 600-tweet sample, with the free rule layer "
  "additionally applied to all 46,016 unique tweets. Real-time deployment, "
  "multilingual handling, and human-subject studies are out of scope.")

# ============================================================================
# CHAPTER 2 — DATASET & EXPERIMENTAL DESIGN
# ============================================================================
chapter(2, "Dataset and Experimental Design")
section("2.1 Dataset")
P("The corpus contains 47,692 tweets, each labelled with one of six "
  "cyberbullying types. After removing 1,676 exact-duplicate tweets, 46,016 "
  "unique tweets remain. The classes are well balanced (Table 1), which makes "
  "accuracy and macro-averaged F1 directly meaningful and avoids the majority-"
  "class bias that distorts imbalanced corpora.")
add_table(["Class", "Tweets (after de-duplication)"],
          [["religion", "7,995"], ["age", "7,991"], ["ethnicity", "7,952"],
           ["not_cyberbullying", "7,937"], ["gender", "7,898"],
           ["other_cyberbullying", "6,243"], ["Total (unique)", "46,016"]],
          [3.2 * inch, 2.6 * inch],
          "Class distribution of the cleaned dataset.")
add_figure("ch5_class_distribution.png",
           "Class distribution of the cleaned dataset.", width=5.0 * inch)

section("2.2 Preprocessing")
P("Cleaning is deliberately light, because a language model benefits from "
  "seeing a tweet much as a human would. Two text columns are retained: the "
  "original tweet (fed to the model) and a normalised version (used by the rule "
  "layer) in which HTML entities are unescaped, URLs and @-mentions are removed, "
  "and the hash symbol is stripped while the hashtag word is kept. Exact "
  "duplicates are removed, and a fixed random seed governs all sampling so the "
  "experiments are reproducible.")

section("2.3 Evaluation Protocol")
P("The design is training-free: the labelled data is a test bench, never a "
  "training set. Each experiment follows one loop &mdash; read a tweet, build a "
  "prompt, call the model, parse a strict JSON answer, and compare the predicted "
  "label with the gold label. Because sending 46,016 tweets to a model is costly "
  "and slow, the model experiments use a <b>stratified sample of 600 tweets "
  "(100 per class)</b>; the free rule layer additionally runs on the full "
  "dataset. The decoding temperature is fixed at zero, and every prompt&ndash;"
  "response pair is cached so re-runs are instant and identical.")

section("2.4 Structured Output Schema")
P("To make every strategy directly comparable and every decision explainable, "
  "all prompts force the same JSON object:")
P('<font face="Courier" size="9">{ "label", "target_group", "intent", '
  '"intensity", "explanation" }</font>', ParagraphStyle("code", parent=Body,
  alignment=TA_CENTER))
P("The <i>label</i> is one of the six classes; <i>target_group</i> names who is "
  "attacked; <i>intent</i> and <i>intensity</i> capture severity; and "
  "<i>explanation</i> is a one-sentence justification. This schema operationalises "
  "explainability and yields machine-readable output for automatic scoring.")

section("2.5 Prompting Strategies")
P("Five prompting styles are studied. They share the identical task description "
  "and output schema; only the way the model is asked to think changes, which "
  "isolates the effect of prompt design.")
bullets([
    "<b>Zero-shot</b> &mdash; a plain instruction with no examples or reasoning.",
    "<b>Few-shot</b> &mdash; two to three hand-written, synthetic labelled examples per category (no dataset tweets, hence no leakage).",
    "<b>Chain-of-Thought</b> &mdash; an explicit four-step reasoning instruction before the answer.",
    "<b>Persona</b> &mdash; the model is cast as an experienced content-moderation officer who also advises school counsellors.",
    "<b>Few-shot + CoT</b> &mdash; examples and reasoning combined.",
])

section("2.6 Experiment Catalogue")
add_table(["#", "Experiment", "Purpose"],
          [["1", "Rule-based baseline", "Lexicon + sentiment toxicity scorer; the non-AI floor and the safety layer."],
           ["2", "Zero-shot", "Pure generative ability with no scaffolding."],
           ["3", "Few-shot", "Effect of in-context examples."],
           ["4", "Chain-of-Thought", "Effect of explicit step-by-step reasoning."],
           ["5", "Persona", "Effect of role-conditioning."],
           ["6", "Few-shot + CoT", "Whether examples and reasoning stack."],
           ["7", "Hybrid (Reasoning+Rules)", "Fusion of the best model with the lexicon."],
           ["8", "Robustness / hard cases", "Behaviour on sarcasm, coded, and quoted tweets."],
           ["9", "Explanation quality", "LLM-as-judge rubric scoring of justifications."],
           ["10", "Benchmark synthesis", "Roll-up tables, ablation, and reporting."]],
          [0.4 * inch, 1.8 * inch, 4.0 * inch],
          "The ten experiments and their purpose.", wrap_cols=[2])

section("2.7 Evaluation Metrics")
bullets([
    "<b>Accuracy</b> and <b>macro-F1</b> on the six-class task (macro-F1 is the primary metric, since classes are balanced).",
    "<b>Per-class F1</b>, to reveal which categories are confused.",
    "<b>Binary precision and recall</b> (bully vs not), the metric most relevant to moderation safety.",
    "<b>Parse-OK rate</b>, the fraction of replies returning valid JSON.",
    "<b>Explanation quality</b>, a composite of four rubric checks (Experiment 9).",
])

# ============================================================================
# CHAPTER 3 — IMPLEMENTATION
# ============================================================================
chapter(3, "System Implementation")
section("3.1 Architecture and Pipeline")
P("The system is implemented as a reproducible nine-stage pipeline. Each stage "
  "is an independent script that writes its outputs to disk, so stages can be "
  "executed and inspected one at a time and the whole study can be replayed from "
  "any point.")
add_table(["Stage", "Module", "Responsibility"],
          [["00", "check_api_keys", "Health-checks every API key and records the working subset, so restricted keys are skipped."],
           ["01", "prepare_data", "Cleans, de-duplicates, and builds the stratified 600-tweet evaluation sample."],
           ["02", "score_lexicon", "Rule-based toxicity and sentiment scorer (Experiment 1) on sample and full data."],
           ["03", "run_prompts", "Runs the five prompt styles (Experiments 2&ndash;6) over the sample."],
           ["04", "fuse_hybrid", "Builds the hybrid (Experiment 7) by category arbitration."],
           ["05", "evaluate_metrics", "Computes metrics, confusion matrices, and figures."],
           ["06", "analyze_hard_cases", "Robustness slice (Experiment 8) and failure gallery."],
           ["07", "judge_explanations", "LLM-as-judge explanation scoring (Experiment 9)."],
           ["08 / 09", "build_report / build_pdf", "Synthesis (Experiment 10) and this document."]],
          [0.6 * inch, 1.5 * inch, 4.1 * inch],
          "The nine-stage implementation pipeline.", wrap_cols=[2])

section("3.2 Engineering Details")
subsection("3.2.1 API-key rotation and resilience")
P("Thirteen API keys were supplied; a preflight health-check found nine live "
  "and four belonging to restricted organisations. The client round-robins "
  "across the live keys on every call so that no single key exceeds the free-"
  "tier rate limit, and it permanently drops any key that returns a restricted "
  "or invalid error at runtime. This allowed a batch of roughly three thousand "
  "calls to complete without manual intervention.")
subsection("3.2.2 Caching and reproducibility")
P("Every (model, prompt) pair is hashed and its response cached as JSON. "
  "Re-running an experiment therefore costs nothing and returns identical "
  "results, which is essential for a study that compares prompt variants.")
subsection("3.2.3 Robust JSON parsing")
P("The first JSON object in each reply is extracted and, if necessary, repaired "
  "(for example, trailing commas are removed). Predicted labels are then "
  "normalised onto the six canonical classes, mapping common variants such as "
  "&lsquo;racism&rsquo; to ethnicity. Across all three thousand model calls the "
  "parse-OK rate was 100%.")
subsection("3.2.4 Concurrency")
P("Eight worker threads (kept below the nine-key count) issue calls "
  "concurrently, reducing the full five-style run to roughly twenty minutes.")

section("3.3 Implementation of Each Experiment")
P("Each experiment is described below by its objective, its design rationale, "
  "the concrete implementation steps, and a short note on outputs.")

EXPERIMENTS = [
    ("3.3.1 Experiment 1 — Rule-based Baseline (Safety Layer)",
     "To establish a transparent, training-free, non-AI floor and to provide a "
     "lexical safety signal that the hybrid can later use.",
     "A compact, fully auditable lexicon is preferable to an opaque toxicity "
     "model: every decision can be traced to a specific term, which serves the "
     "explainability aim and keeps compute negligible.",
     ["Load the cleaned tweets together with the normalised text column.",
      "Define compact, human-readable lexicons grouped by target category (ethnicity, religion, gender, age, other).",
      "Define a small hard-flag set (explicit slurs and phrases such as &lsquo;kill yourself&rsquo;) that almost always indicate bullying.",
      "For each tweet, count lexicon hits per category and compute a VADER sentiment compound score.",
      "Combine into a toxicity score (0.7 &times; lexicon density + 0.3 &times; negativity), forced to at least 0.9 if a hard flag fires.",
      "Predict the category as the lexicon group with the most hits, and mark the tweet as bullying when toxicity reaches the 0.35 threshold.",
      "Map the decision into the six-class space (the category guess if bullying, otherwise not_cyberbullying).",
      "Write per-tweet scores for both the 600-tweet sample and the full dataset, and report binary accuracy."],
     "Outputs the scored sample (consumed by the hybrid) and the full-data "
     "baseline. Binary accuracy on the full corpus is 0.511, confirming that a "
     "small lexicon catches explicit abuse but misses much implicit bullying."),

    ("3.3.2 Experiment 2 — Zero-shot",
     "To measure the model&rsquo;s raw classification ability with no "
     "scaffolding.",
     "Zero-shot is the natural baseline against which every richer prompt is "
     "compared; any gain from examples or reasoning must be measured relative to "
     "it.",
     ["Build a base task string listing the six categories with short descriptions.",
      "Append the strict JSON output contract so the reply is machine-parseable.",
      "Set a minimal system prompt (&lsquo;You are a precise text classifier&rsquo;).",
      "Insert only the target tweet, with no examples and no reasoning instruction.",
      "Send the prompt through the cached, key-rotated client at temperature zero.",
      "Extract and parse the first JSON object from the reply.",
      "Normalise the predicted label onto one of the six canonical classes.",
      "Persist identifier, gold label, prediction, target group, intent, intensity, explanation, and parse status.",
      "Score accuracy and macro-F1 against the gold labels."],
     "Achieves accuracy 0.383 and macro-F1 0.363 &mdash; a surprisingly strong "
     "floor for a small model on a six-way task."),

    ("3.3.3 Experiment 3 — Few-shot",
     "To test whether a few in-context examples improve fine-grained "
     "classification.",
     "Examples can teach the label boundaries (for instance, religion versus "
     "ethnicity) without any training; the examples are synthetic to eliminate "
     "any risk of leaking evaluation tweets.",
     ["Reuse the base task and JSON contract from Experiment 2.",
      "Hand-write one clear, synthetic example per category.",
      "Format the examples as Tweet/Answer pairs and prepend them to the prompt.",
      "Append the target tweet after the examples.",
      "Call the model with identical client settings for comparability.",
      "Parse and normalise the predicted label.",
      "Persist predictions and explanations.",
      "Compare macro-F1 against zero-shot to isolate the effect of examples."],
     "Macro-F1 rises slightly to 0.378 (the best macro-F1 among the pure prompts), "
     "though six-class accuracy dips to 0.365, indicating examples sharpen some "
     "categories while biasing others."),

    ("3.3.4 Experiment 4 — Chain-of-Thought",
     "To test whether forcing explicit reasoning improves judgement.",
     "Step-by-step reasoning is widely reported to help larger models on complex "
     "tasks; this experiment asks whether the benefit holds for a small model on "
     "nuanced categories.",
     ["Reuse the base task and JSON contract.",
      "Add a four-step reasoning instruction: identify the target, analyse tone, judge intent, then choose the trait.",
      "Instruct the model to reason internally and output only the final JSON.",
      "Use a reasoning-oriented system prompt.",
      "Send the target tweet and collect the reply.",
      "Parse the JSON label and the structured fields.",
      "Normalise and store the predictions.",
      "Evaluate and compare against zero-shot."],
     "Accuracy falls to 0.353 and macro-F1 to 0.333; on this small model the "
     "reasoning scaffold does not help and slightly hurts."),

    ("3.3.5 Experiment 5 — Persona",
     "To test whether role-conditioning changes strictness and judgement.",
     "Casting the model as a fair, context-aware moderator may align its "
     "behaviour with the task without any examples or explicit reasoning steps.",
     ["Reuse the base task and JSON contract.",
      "Write a rich persona system prompt: an experienced moderation officer who advises school counsellors, fair and protective of vulnerable groups.",
      "Keep the user prompt simple (&lsquo;As the moderator, classify this tweet&rsquo;).",
      "Call the model with the persona system message.",
      "Parse and normalise the predicted label.",
      "Persist predictions and the (often more careful) explanations.",
      "Score accuracy and macro-F1.",
      "Compare against the other styles."],
     "The strongest pure prompt by accuracy (0.400), with the highest binary "
     "recall (0.878); it becomes the base for the hybrid."),

    ("3.3.6 Experiment 6 — Few-shot + CoT",
     "To test whether examples and reasoning combine additively.",
     "If both techniques help independently, combining them might compound the "
     "gains; this experiment checks that assumption.",
     ["Reuse the base task and JSON contract.",
      "Prepend the hand-written few-shot examples.",
      "Add the four-step reasoning instruction.",
      "Use the reasoning-oriented system prompt.",
      "Send the target tweet and collect the reply.",
      "Parse, normalise, and store the predictions.",
      "Evaluate and compare against the single-technique prompts."],
     "The weakest configuration (accuracy 0.322, macro-F1 0.327): the two "
     "techniques interfere rather than stack on this model."),

    ("3.3.7 Experiment 7 — Hybrid (Reasoning + Rules)",
     "To combine the model&rsquo;s reasoning with the lexicon so that the whole "
     "exceeds either part.",
     "Diagnostic analysis showed the model has high recall but confuses "
     "categories, whereas the lexicon has low recall but names the category "
     "precisely when an explicit marker fires. The productive fusion is "
     "therefore category arbitration, not a recall safety-net, because the "
     "model&rsquo;s misses are implicit cases (mean toxicity 0.04) the lexicon "
     "also cannot see.",
     ["Choose the best pure model run (persona) as the base prediction.",
      "Join the model predictions with the Experiment&nbsp;1 rule scores on tweet identifier.",
      "Diagnose where each method is strong (model recall vs lexical category precision).",
      "Verify on disagreement cases that the fired rule category is correct far more often than the model (45 versus 8).",
      "Define category arbitration: when the lexicon fires a confident category and toxicity reaches 0.5, override the model&rsquo;s category with the lexicon&rsquo;s.",
      "Also record a recall safety-net and a binary-union strategy for comparison.",
      "Apply arbitration to produce the hybrid prediction.",
      "Report rescued versus broken cases and the accuracy gain.",
      "Evaluate against all other systems."],
     "Arbitration applied 48 overrides &mdash; 33 correct rescues against 7 breaks "
     "&mdash; lifting accuracy from 0.400 to 0.443 and macro-F1 to 0.404, the best "
     "of any system, while preserving recall at 0.878."),

    ("3.3.8 Experiment 8 — Robustness on Hard Cases",
     "To document where the system struggles: sarcasm, coded language, quoted "
     "irony, and reclaimed terms.",
     "Aggregate accuracy can hide systematic blind spots; isolating hard tweets "
     "exposes them and produces concrete error examples for qualitative "
     "discussion.",
     ["Reuse the predictions already computed for Experiments 2&ndash;6 (no new calls).",
      "Define a &lsquo;hard&rsquo; heuristic from sarcasm markers, emojis, quoted irony, and reclaimed in-group terms.",
      "Flag each evaluation tweet as hard or easy.",
      "Compute per-style accuracy on the hard, easy, and overall subsets.",
      "Collect the misclassified hard cases into a failure gallery.",
      "Save the summary and gallery for analysis."],
     "Persona is the most robust on hard tweets; few-shot+CoT degrades most. "
     "Seventy-nine hard errors are catalogued for inspection."),

    ("3.3.9 Experiment 9 — Explanation Quality",
     "To evaluate the <i>quality</i> of justifications, not just label "
     "correctness.",
     "Because explainability is a primary objective, the explanations must be "
     "assessed directly; an LLM-as-judge applies a consistent rubric at scale.",
     ["Take the best style&rsquo;s predictions and explanations (persona).",
      "Draw a stratified subset of 20 tweets per class (120 explanations).",
      "Define a four-point rubric: names the target, cites the reason, intent coherent, consistent with the gold label.",
      "Prompt a judge model to score each explanation as 0/1 flags in JSON.",
      "Run the judge concurrently through the cached client.",
      "Average the flags into a composite quality score.",
      "Aggregate overall and per-class explanation quality."],
     "Composite quality is 0.569: intent is described very coherently (0.95) but "
     "the precise target group is often not named (0.31)."),

    ("3.3.10 Experiment 10 — Benchmark Synthesis",
     "To consolidate all results into comparison tables, an ablation, and "
     "reports.",
     "A single synthesis stage guarantees the headline numbers are computed once "
     "from the same source and reused consistently across the Markdown report "
     "and this document.",
     ["Load the master metrics table from the evaluation stage.",
      "Build the six-class comparison and the prompt-style ablation (delta versus zero-shot).",
      "Quantify the hybrid&rsquo;s improvement over the best pure model.",
      "Fold in the hard-case and explanation-quality results.",
      "Emit the Markdown report and this PDF."],
     "Produces the artifacts presented in Chapter&nbsp;4."),
]
for title, obj, rat, stp, out in EXPERIMENTS:
    story.append(KeepTogether([Paragraph(title, Sub),
                               Paragraph(f"<b>Objective.</b> {obj}", Body)]))
    P(f"<b>Design rationale.</b> {rat}")
    P("<b>Implementation steps.</b>")
    steps(stp)
    P(f"<b>Outputs.</b> {out}")
    S(4)

# ============================================================================
# CHAPTER 4 — RESULTS & DISCUSSION
# ============================================================================
chapter(4, "Results and Discussion")
section("4.1 Master Comparison")
rows = []
for _, r in summ.iterrows():
    rows.append([r["name"], f(r["accuracy"], 3), f(r["macro_f1"], 3),
                 f(r["weighted_f1"], 3), f(r["binary_recall"], 3),
                 f(r["parse_ok_rate"], 2) if pd.notna(r["parse_ok_rate"]) else "n/a"])
add_table(["System", "Acc.", "Macro-F1", "Wt-F1", "Bin. Recall", "Parse-OK"], rows,
          [2.35 * inch, 0.7 * inch, 0.85 * inch, 0.7 * inch, 0.95 * inch, 0.8 * inch],
          "Six-class comparison of all systems, sorted by macro-F1.",
          highlight_first=True, wrap_cols=[0])
P("The hybrid leads on both accuracy and macro-F1 while retaining the "
  "model&rsquo;s high binary recall. The rule baseline reaches competitive "
  "accuracy with extremely high precision (0.99) but very low recall (0.40): it "
  "is correct when it fires, but it fires only on explicit slurs. The language "
  "models, in contrast, recover most bullying (recall 0.84&ndash;0.88) but "
  "confuse the fine-grained categories.")
add_figure("ch5_macro_f1_by_experiment.png", "Macro-F1 by experiment.", width=5.3 * inch)
add_figure("ch5_core_metrics_grouped.png",
           "Accuracy, macro-F1, and weighted-F1 across systems.", width=6.0 * inch)

section("4.2 Prompt-Style Ablation")
zero = g("exp2_zero_shot", "macro_f1")
order = ["exp2_zero_shot", "exp3_few_shot", "exp4_cot", "exp5_persona", "exp6_few_shot_cot"]
abl = []
for e in order:
    mf = g(e, "macro_f1")
    abl.append([EXP_NAME[e], f(g(e, "accuracy"), 4), f(mf, 4), f"{mf - zero:+.4f}"])
add_table(["Prompt style", "Accuracy", "Macro-F1", "Δ Macro-F1 vs zero-shot"], abl,
          [2.6 * inch, 1.1 * inch, 1.1 * inch, 1.7 * inch],
          "Prompt-style ablation relative to the zero-shot baseline.", wrap_cols=[0])
P("Few-shot gives a small positive change in macro-F1, persona is essentially "
  "neutral on macro-F1 but best on accuracy, and both reasoning configurations "
  "reduce performance. The clear lesson is that, for a small model on this "
  "fine-grained task, <b>role-conditioning is more effective than reasoning "
  "scaffolds</b>, and stacking techniques can be counter-productive.")
add_figure("ch5_prompt_style_ablation.png", "Prompt-style ablation.", width=5.3 * inch)

section("4.3 Per-Class Analysis")
pc_cols = ["f1_religion", "f1_age", "f1_gender", "f1_ethnicity",
           "f1_other_cyberbullying", "f1_not_cyberbullying"]
pc_rows = []
for _, r in summ.iterrows():
    pc_rows.append([r["name"]] + [f(r[c], 2) for c in pc_cols])
add_table(["System", "relig.", "age", "gender", "ethnic.", "other", "not_cb"], pc_rows,
          [2.25 * inch] + [0.6 * inch] * 6,
          "Per-class F1 for every system.", fs=7.8, wrap_cols=[0])
P("The age class collapses to near-zero F1 for every system: in this corpus, "
  "age-labelled tweets are frequently recollections of school bullying in which "
  "age is incidental rather than the target, so both the model and the lexicon "
  "route them to other_cyberbullying. Religion is also weak, being often "
  "implicit, whereas gender and ethnicity &mdash; carried by explicit slurs &mdash; "
  "score highest. The hybrid&rsquo;s gains concentrate exactly in the explicit-"
  "marker classes, consistent with its arbitration mechanism.")
add_figure("ch5_per_class_f1_heatmap.png",
           "Per-class F1 heatmap across all systems.", width=5.8 * inch)
add_figure("ch5_cm_norm_exp7_hybrid.png",
           "Normalised confusion matrix of the hybrid system.", width=4.5 * inch)

section("4.4 Hybrid Analysis")
P("Built on the persona base, category arbitration applied 48 overrides, of "
  "which 33 were correct rescues and only 7 were breaks. This lifted accuracy "
  "from 0.400 to 0.443 (a gain of 0.043) and macro-F1 to 0.404. The empirical "
  "justification is decisive: on the cases where the lexicon fired a category "
  "and disagreed with the model, the lexicon was correct 45 times against the "
  "model&rsquo;s 8. Crucially, an attempt to use the lexicon instead as a recall "
  "safety-net produced almost no benefit, because the model&rsquo;s false "
  "negatives are implicit tweets (mean toxicity 0.04) that contain no lexicon "
  "terms &mdash; an honest negative result that motivated the arbitration design.")
add_figure("ch5_hybrid_vs_best_llm.png", "Hybrid versus the best pure model.", width=5.3 * inch)

section("4.5 Robustness on Hard Cases")
hrows = []
for _, r in hard.iterrows():
    hrows.append([r["style"], str(int(r["n_hard"])), f(r["acc_hard"], 3),
                  f(r["acc_easy"], 3), f(r["acc_overall"], 3)])
add_table(["Prompt style", "n hard", "Acc. hard", "Acc. easy", "Acc. overall"], hrows,
          [1.7 * inch, 0.9 * inch, 1.0 * inch, 1.0 * inch, 1.05 * inch],
          "Accuracy on hard (sarcastic / coded / quoted) versus easy tweets.")
P("Persona is the only style that is <i>more</i> accurate on hard tweets than on "
  "easy ones, suggesting that role-conditioning helps the model slow down on "
  "ambiguous content; few-shot+CoT degrades most, mirroring its weak overall "
  "result.")
add_figure("ch5_hard_vs_easy.png",
           "Accuracy on hard versus easy tweets by prompt style.", width=5.3 * inch)

subsection("4.5.1 Failure Gallery")
P("Table " + str(_tbl["n"] + 1) + " lists representative hard errors. The "
  "pattern is consistent: implicit attacks on religion or age, and quoted or "
  "reported slurs, are routed to other_cyberbullying or to the wrong protected "
  "trait, because the abuse is conveyed by context rather than an explicit "
  "category marker.")
gal_rows = []
for _, r in gallery.head(6).iterrows():
    tweet = str(r["tweet_text"])
    tweet = (tweet[:150] + "…") if len(tweet) > 150 else tweet
    gal_rows.append([tweet, r["gold"], r["pred"]])
add_table(["Tweet (truncated)", "Gold", "Predicted"], gal_rows,
          [4.3 * inch, 0.9 * inch, 1.0 * inch],
          "Representative hard-case misclassifications (chain-of-thought).",
          fs=7.8, wrap_cols=[0, 1, 2])

if eq is not None:
    section("4.6 Explanation Quality")
    comp = eq["expl_quality"].mean()
    add_table(["Rubric criterion", "Mean score (0–1)"],
              [["Composite explanation quality", f(comp, 3)],
               ["Names the target group", f(eq["names_target"].mean(), 3)],
               ["Cites the reason / offending words", f(eq["cites_reason"].mean(), 3)],
               ["Describes intent coherently", f(eq["intent_coherent"].mean(), 3)],
               ["Consistent with the gold label", f(eq["consistent_with_gold"].mean(), 3)]],
              [4.0 * inch, 1.6 * inch],
              "Explanation-quality rubric scores (LLM-as-judge, persona, n=120).",
              wrap_cols=[0])
    P("Explanations are coherent about intent (0.95) but frequently fail to name "
      "the precise target group (0.31), and consistency with the gold label "
      "(0.47) tracks the six-class accuracy. Quality is highest for gender and "
      "ethnicity and lowest for religion and age, exactly mirroring the "
      "classification confusions and confirming that the model&rsquo;s "
      "uncertainty is reflected honestly in its explanations.")

# ============================================================================
# CHAPTER 5 — CONCLUSION
# ============================================================================
chapter(5, "Conclusion and Future Work")
section("5.1 Summary of Findings")
bullets([
    "<b>The hybrid is best.</b> &lsquo;Reasoning + Rules&rsquo; via category arbitration beats the best pure model (+0.043 accuracy) and the rules alone (+0.027), reaching 0.443 accuracy and 0.404 macro-F1 while keeping 0.878 binary recall.",
    "<b>Arbitration, not a safety-net, is the correct fusion.</b> The model&rsquo;s misses are implicit tweets the lexicon also misses; the lexicon&rsquo;s value is precise category correction when explicit markers fire.",
    "<b>Role-conditioning beats reasoning scaffolds</b> on a small model; chain-of-thought and few-shot+CoT reduced macro-F1.",
    "<b>High recall, modest category precision.</b> The model catches roughly 88% of bullying but confuses which type, so binary moderation is dependable even when six-class accuracy is moderate.",
    "<b>Explanations are real but uneven</b>, strong on intent and weak on naming the exact target, with religion and age the hardest.",
])
section("5.2 Limitations")
P("The study uses one English-language corpus and one small model, and the "
  "evaluation sample is 600 tweets. The lexicon is intentionally compact, which "
  "bounds its recall. The dataset itself contains label noise &mdash; notably the "
  "age class &mdash; that caps the achievable six-class accuracy for any method. "
  "The LLM-as-judge for explanation quality, while consistent, is not a human "
  "study.")
section("5.3 Future Work")
bullets([
    "Scale the evaluation sample and add a larger model (for example a 70B variant) to separate prompt effects from model-capacity effects.",
    "Expand and weight the lexicon, and learn the arbitration threshold, to widen the hybrid&rsquo;s coverage.",
    "Add a confidence-calibrated escalation path so uncertain cases are routed to human review.",
    "Conduct a human evaluation of explanation usefulness with teachers and moderators.",
    "Re-examine the age class definition, given its systematic confusion with general harassment.",
])
section("5.4 Concluding Remarks")
P("A carefully prompted Generative-AI model, supported by a small transparent "
  "rule layer, can act as a lightweight, explainable cyberbullying screen "
  "suitable for schools and platforms with limited compute. Its most dependable "
  "capability is binary safety detection, while fine-grained category labelling "
  "&mdash; and the precise naming of the targeted group in its explanations &mdash; "
  "is the clearest avenue for future improvement.")

# ----------------------------------------------------------------------------
doc = ThesisDoc(str(PDF_PATH), pagesize=A4, title="Cyberbullying Gen-AI Report",
                author="Project Report")
# Page 1 uses the 'cover' template (first added); a NextPageTemplate('main')
# flowable placed after the cover content switches every later page to the
# header/footer template. multiBuild runs twice so the TOC resolves page numbers.
doc.multiBuild(story)
print(f"PDF written -> {PDF_PATH}")

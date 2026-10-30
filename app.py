# ==============================================================
# 1. IMPORTS
# ==============================================================
import io
import json
import re
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==============================================================
# 2. CONFIG (st.set_page_config must be the first Streamlit command)
# ==============================================================
st.set_page_config(
    page_title="PG Finder: Smart PG Review Analyst",
    page_icon=":material/apartment:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The two Small Language Models the user can pick from
SLM_OPTIONS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# The five aspects the SLM scores for every review
ASPECTS = ["food", "safety", "wifi", "cleanliness", "owner"]
ASPECT_LABELS = {"food": "Food", "safety": "Safety", "wifi": "Wi-Fi", "cleanliness": "Cleanliness", "owner": "Owner"}

# Exact system prompts for the two SLM tasks
ASPECT_SYSTEM_PROMPT = 'Read this PG (paying guest) review from Bengaluru. Respond ONLY with JSON, scores 1-5 or null if not mentioned: {"food": , "safety": , "wifi": , "cleanliness": , "owner": , "summary": "max 12 words"}'
RECO_SYSTEM_PROMPT = "You are a helpful PG advisor in Bengaluru. Using ONLY the PG data given and the user's needs, recommend the best PG and one backup. Explain why in max 4 short bullet points. Do not invent facts."

# Which PG areas are a sensible commute from each office area (simple rule-based filter)
OFFICE_TO_PG_AREAS = {
    "Whitefield": ["Marathahalli"],
    "Marathahalli": ["Marathahalli"],
    "Bellandur / Sarjapur Road": ["Marathahalli", "BTM"],
    "Koramangala / HSR / Silk Board": ["BTM"],
    "Electronic City": ["Electronic City"],
    "Anywhere in Bengaluru": [],
}

# Colours used in charts (kept in one place)
COLOR_POS = "#3ddc97"
COLOR_NEU = "#8b949e"
COLOR_NEG = "#f85149"
LINE_COLORS = ["#3ddc97", "#4ea1ff", "#f2cc60"]

# Small CSS layer: monospace headings, compact spacing, hide default chrome
st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
h1, h2, h3 {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; letter-spacing: -0.02em;}
h1 {font-size: 1.7rem !important; font-weight: 600 !important;}
h2, h3 {font-size: 1.05rem !important; font-weight: 600 !important;}
div[data-testid="stMetricValue"] {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 1.5rem;}
div[data-testid="stMetricLabel"] {text-transform: uppercase; font-size: 0.68rem; letter-spacing: 0.08em; opacity: 0.7;}
div[data-testid="stCaptionContainer"] {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.78rem;}
header[data-testid="stHeader"] {background: transparent;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================
# 3. MODEL LOADING (all cached, all lazy)
# ==============================================================
@st.cache_resource(show_spinner=False)
def load_slm(model_name):
    # Loads the chosen Small Language Model once and keeps it in memory
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_embedder():
    # Sentence embedding model used for semantic search
    import torch
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_vader():
    # NLTK VADER: a classic rule-based sentiment scorer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


@st.cache_data(show_spinner=False)
def embed_texts(texts):
    # texts MUST be a tuple of strings (so Streamlit can hash it); the model is loaded inside
    model = load_embedder()
    return model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)

# ==============================================================
# 4. HELPERS
# ==============================================================
def get_device_label():
    # Shows which hardware the models will run on
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda (" + torch.cuda.get_device_name(0) + ")"
        return "cpu"
    except Exception:
        return "cpu"


def mark_loaded(name):
    # Remembers which models have been loaded in this session (for the sidebar panel)
    if name not in st.session_state.loaded:
        st.session_state.loaded.append(name)


def safe_float(value, default):
    # Turns anything into a float, or returns the default if that is not possible
    try:
        return float(value)
    except Exception:
        return default


def as_list(value):
    # Wraps a single value into a list (and keeps lists as they are)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def slm_chat(system_prompt, user_prompt, max_new_tokens=256):
    # Sends a system + user prompt to the SLM and returns its text answer
    import torch
    tokenizer, model = load_slm(st.session_state.model_name)
    mark_loaded(st.session_state.model_name)
    user_prompt = user_prompt[:6000]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_json(text):
    # Pulls the first {...} block out of the SLM text and parses it; None if anything goes wrong
    try:
        cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(0))
    except Exception:
        return None


def slm_json(system_prompt, user_prompt, max_new_tokens=300):
    # Asks the SLM for JSON; retries once with a stricter reminder; returns dict or None
    text = slm_chat(system_prompt, user_prompt, max_new_tokens)
    data = parse_json(text)
    if data is None:
        text = slm_chat(system_prompt, user_prompt + "\nReturn ONLY valid JSON. No extra text.", max_new_tokens)
        data = parse_json(text)
    if isinstance(data, dict):
        return data
    return None


def show_prompts(system_prompt, user_prompt):
    # Lets learners study exactly what was sent to the SLM
    with st.expander("See what we asked the SLM"):
        st.markdown("**System prompt**")
        st.code(system_prompt, language=None)
        st.markdown("**User prompt**")
        st.code(user_prompt, language=None)


def set_status(name, kind, message):
    # Stores a message in session_state so it stays visible after reruns
    st.session_state.status[name] = (kind, message)


def show_status(name):
    item = st.session_state.status.get(name)
    if not item:
        return
    kind, message = item
    if kind == "error":
        st.error(message)
    elif kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    else:
        st.info(message)


def vader_label(compound):
    # Label rule from the brief
    if compound > 0.05:
        return "Positive"
    if compound < -0.05:
        return "Negative"
    return "Neutral"


def mentions_veg(text):
    # True if a review says something positive about vegetarian food
    t = str(text).lower()
    t = re.sub(r"non[- ]?veg(etarian)?", " ", t)
    if re.search(r"\b(no|not|limited|few|hardly any)\s+(pure\s+)?veg", t):
        return False
    return bool(re.search(r"\bveg\b|vegetarian|\bjain\b", t))


def mentions_nonveg(text):
    # True if a review mentions non-vegetarian food
    t = str(text).lower()
    return bool(re.search(r"non[- ]?veg|chicken|\beggs?\b|\bfish\b|mutton", t))


def normalize_gender(value):
    # Maps many spellings to ladies / gents / unisex
    s = str(value).strip().lower()
    if any(w in s for w in ["lad", "women", "female", "girl"]):
        return "ladies"
    if any(w in s for w in ["gent", "men", "male", "boy"]):
        return "gents"
    return "unisex"


def money(value):
    try:
        return "₹" + format(int(value), ",")
    except Exception:
        return str(value)


def style_fig(fig):
    # One place for chart layout so all charts look the same
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="", title_font_size=14)
    return fig

# ==============================================================
# 5. SAMPLE DATA (all names, numbers and people are made up)
# ==============================================================
SAMPLE_PGS = [
    {
        "pg_name": "Sri Sai Comforts", "area": "Marathahalli", "rent": 9500, "gender": "ladies",
        "reviews": [
            "Food is homely and there is always a pure veg option. The rotis are soft and the Sunday special is tasty.",
            "Very safe for girls. A guard sits at the gate all night and there are CCTV cameras in every corridor.",
            "Wi-Fi is fast and steady, I attend video calls from my room without any lag.",
            "Rooms and bathrooms are cleaned daily and the sheets are changed every week. Very hygienic.",
            "The owner, Mrs. Revathi, is polite and fixes any problem within a day.",
            "Curfew is 10 pm on weekdays which is a bit strict, but guests are allowed in the visitors hall.",
            "Rent is slightly high for a shared room but it includes meals, laundry and a geyser.",
            "Dinner is sometimes cold if you come back late, otherwise a very comfortable PG.",
        ],
    },
    {
        "pg_name": "Green Nest PG", "area": "BTM 2nd Stage", "rent": 8000, "gender": "gents",
        "reviews": [
            "Food is okay, chicken on weekends and egg curry twice a week. Only a few veg dishes, mostly the same dal.",
            "Wi-Fi is very slow in the evenings and keeps disconnecting. Cannot do video calls.",
            "No curfew at all, you can come back at any time. Great for people with night shifts.",
            "Security is average. There is a gate lock but no guard and no CCTV.",
            "Bathrooms are cleaned only twice a week and smell bad sometimes.",
            "The owner, Mr. Prakash, is friendly and lets us pay rent a few days late.",
            "Location is great, five minutes from Silk Board and lots of restaurants nearby.",
            "Rent of 8000 is fair for a double sharing room with a cupboard and a fan.",
        ],
    },
    {
        "pg_name": "Lakshmi Ladies Residency", "area": "Marathahalli", "rent": 8500, "gender": "ladies",
        "reviews": [
            "Pure veg kitchen, the food is fresh and no onion garlic on Mondays. Best homely food I got in Bengaluru.",
            "Safe and calm, a warden stays on the premises and the main door is locked at 9.30 pm.",
            "Wi-Fi works but is slow when everyone is online at night. Fine for browsing, not for big downloads.",
            "Very clean rooms, the housekeeping lady comes twice a day.",
            "The owner is strict about rules and sometimes rude if you return late.",
            "Curfew of 9.30 pm is too early for working girls with long commutes.",
            "Rent of 8500 is reasonable and includes breakfast and dinner.",
            "Quiet place to study or sleep, no loud music is allowed after 9 pm.",
        ],
    },
    {
        "pg_name": "TechNest Co-Living", "area": "Electronic City Phase 1", "rent": 10500, "gender": "unisex",
        "reviews": [
            "Wi-Fi is excellent, a dedicated 100 Mbps line and a quiet work lounge. Perfect for work from home.",
            "Food has both veg and non-veg counters and the menu changes daily. Tasty but a little oily.",
            "Safety is good, biometric entry and a guard at the gate. Girls and boys floors are separate.",
            "Rooms are modern and cleaned daily, with a laundry room on every floor.",
            "The manager, Mr. Karthik, replies to complaints on WhatsApp within an hour.",
            "No curfew but guests need a prior entry at the reception.",
            "Rent of 10500 is costly for students but fair for working people.",
            "Free shuttle to the nearby tech parks on weekdays saves a lot of money.",
        ],
    },
    {
        "pg_name": "Amma's Home PG", "area": "BTM 1st Stage", "rent": 7000, "gender": "ladies",
        "reviews": [
            "Aunty cooks like home, vegetarian meals with curd rice and sambar. Non-veg only on request.",
            "Feels safe, the owner family lives on the ground floor and knows everyone.",
            "Wi-Fi is weak on the second floor and often down on weekends.",
            "Rooms are old but clean, the bathrooms are cleaned every morning.",
            "The owner, Mrs. Geetha, treats us like her daughters, very caring.",
            "Curfew is 10.30 pm and she calls parents if someone is late.",
            "Cheapest PG in the area at 7000 with food included.",
            "Water supply is good but hot water is available only in the morning.",
        ],
    },
    {
        "pg_name": "Metro Stay Gents PG", "area": "Electronic City Phase 2", "rent": 6500, "gender": "gents",
        "reviews": [
            "Food is bad and oily, there is no veg menu at all, only egg or chicken.",
            "Rooms are dirty and the corridor is never swept. Cockroaches in the kitchen.",
            "Wi-Fi is not provided properly, the router is in the owner's office and only works near it.",
            "Safety is a worry, anyone can walk in as there is no guard at night.",
            "The owner, Mr. Naidu, is rude and refuses to return the deposit on time.",
            "Very cheap at 6500 and the bus stop for tech parks is right outside.",
            "No curfew, but the main gate is locked at midnight.",
            "Power cuts are frequent and there is no inverter backup.",
        ],
    },
    {
        "pg_name": "Sunrise Premium PG", "area": "Marathahalli", "rent": 12000, "gender": "unisex",
        "reviews": [
            "Food is top class, a chef cooks fresh meals with veg, Jain and non-veg options every day.",
            "Excellent safety, smart door locks, CCTV and a 24x7 guard. Separate wings for girls and boys.",
            "Wi-Fi is superfast and every room gets its own router.",
            "Housekeeping is professional, the rooms are spotless and the AC works well.",
            "The owner, Mr. Sandeep, is professional and the staff are helpful.",
            "Rent of 12000 is expensive, but you get a gym, a TV lounge and power backup.",
            "No curfew for working people, just inform the warden if you will be very late.",
            "Whitefield and Bellandur offices are both within 30 minutes by cab.",
        ],
    },
    {
        "pg_name": "Blue Orchid Women's PG", "area": "BTM 2nd Stage", "rent": 9000, "gender": "ladies",
        "reviews": [
            "Very safe, a woman warden stays inside, CCTV everywhere and entry is by fingerprint.",
            "Food is average, the veg thali is repeated too often and the taste is bland.",
            "Wi-Fi is decent for emails and calls but the speed drops after 11 pm.",
            "Clean and well maintained, a cleaner mops the rooms every day.",
            "The owner is helpful and arranged a plumber the same day.",
            "Curfew is 10 pm and the warden is strict but it feels fair for safety.",
            "Rent of 9000 is okay, a bit high for the food quality.",
            "Close to Silk Board metro and many offices, easy commute.",
        ],
    },
]


def build_sample_df():
    # Flattens the sample PGs into one row per review
    rows = []
    for pg in SAMPLE_PGS:
        for review in pg["reviews"]:
            rows.append({
                "pg_name": pg["pg_name"], "area": pg["area"], "rent": pg["rent"],
                "gender": pg["gender"], "review": review,
            })
    return pd.DataFrame(rows)

# ==============================================================
# 6. NLP FUNCTIONS
# ==============================================================
def get_dataframe(uploaded):
    # Returns (dataframe, source text). Falls back to the sample data if the CSV is not usable
    sample = build_sample_df()
    if uploaded is None:
        return sample, "sample (8 PGs, 64 reviews)"
    try:
        required = ["pg_name", "area", "rent", "gender", "review"]
        raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        clean = raw[required].copy()
        clean = clean.dropna(subset=["pg_name", "review"])
        clean["pg_name"] = clean["pg_name"].astype(str).str.strip()
        clean["area"] = clean["area"].astype(str).str.strip()
        clean["review"] = clean["review"].astype(str).str.strip()
        clean["rent"] = pd.to_numeric(clean["rent"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
        before = len(clean)
        clean = clean.dropna(subset=["rent"])
        clean = clean[(clean["pg_name"] != "") & (clean["review"] != "")]
        if len(clean) < before:
            st.warning("Some rows had no readable rent and were skipped.")
        clean["gender"] = clean["gender"].apply(normalize_gender)
        clean = clean.reset_index(drop=True)
        if clean.empty:
            raise ValueError("The file has no usable rows.")
        return clean, "upload (" + str(clean["pg_name"].nunique()) + " PGs, " + str(len(clean)) + " reviews)"
    except Exception as err:
        st.error("Could not read the CSV (" + str(err)[:150] + "). Falling back to sample data.")
        return sample, "sample (8 PGs, 64 reviews)"


def build_pg_table(df):
    # One row per PG with VADER sentiment and simple food flags
    vader = load_vader()
    work = df.copy()
    work["compound"] = work["review"].apply(lambda t: vader.polarity_scores(str(t))["compound"])
    work["veg_flag"] = work["review"].apply(mentions_veg)
    work["nonveg_flag"] = work["review"].apply(mentions_nonveg)
    table = work.groupby("pg_name", sort=False).agg(
        area=("area", "first"),
        rent=("rent", "first"),
        gender=("gender", "first"),
        reviews=("review", "count"),
        vader=("compound", "mean"),
        veg_friendly=("veg_flag", "any"),
        nonveg_friendly=("nonveg_flag", "any"),
    ).reset_index()
    table["sentiment"] = table["vader"].apply(vader_label)
    return table


def pg_display(table, columns):
    # Prepares a table for display: picks columns and capitalises the gender column
    out = table[columns].copy()
    if "gender" in out.columns:
        out["gender"] = out["gender"].astype(str).str.title()
    return out


# Shared column formatting for st.dataframe
PG_COLUMN_CONFIG = {
    "pg_name": st.column_config.TextColumn("PG"),
    "area": st.column_config.TextColumn("Area"),
    "rent": st.column_config.NumberColumn("Rent (INR)", format="%d"),
    "gender": st.column_config.TextColumn("For"),
    "reviews": st.column_config.NumberColumn("Reviews", format="%d"),
    "vader": st.column_config.ProgressColumn("VADER", min_value=-1.0, max_value=1.0, format="%.2f"),
    "sentiment": st.column_config.TextColumn("Label"),
    "reviews_scored": st.column_config.NumberColumn("Scored", format="%d"),
    "food": st.column_config.NumberColumn("Food", format="%.2f"),
    "safety": st.column_config.NumberColumn("Safety", format="%.2f"),
    "wifi": st.column_config.NumberColumn("Wi-Fi", format="%.2f"),
    "cleanliness": st.column_config.NumberColumn("Cleanliness", format="%.2f"),
    "owner": st.column_config.NumberColumn("Owner", format="%.2f"),
}


def score_review(review):
    # Asks the SLM for aspect scores of ONE review; returns a clean dict or None
    user_prompt = "Review: " + review
    st.session_state.last_aspect_prompt = (ASPECT_SYSTEM_PROMPT, user_prompt)
    data = slm_json(ASPECT_SYSTEM_PROMPT, user_prompt, max_new_tokens=120)
    if data is None:
        return None
    result = {}
    for aspect in ASPECTS:
        value = safe_float(data.get(aspect), None)
        if value is None or value != value:
            result[aspect] = None
        else:
            result[aspect] = min(max(value, 1.0), 5.0)
    summary = " ".join(str(x) for x in as_list(data.get("summary", "")))
    result["summary"] = " ".join(summary.split()[:12])
    return result


def run_aspect_scoring(reviews, status_name):
    # Scores every review that is not scored yet by the current model. Returns True if the SLM ran fine
    model_name = st.session_state.model_name
    cache = st.session_state.aspect_cache
    todo = [r for r in dict.fromkeys(reviews) if (model_name, r) not in cache]
    if not todo:
        set_status(status_name, "success", "cache hit: all selected reviews already scored by " + model_name)
        return True
    failed = 0
    started = time.time()
    progress = st.progress(0.0)
    try:
        with st.spinner("Thinking with the SLM... (first run downloads the model, so it can take a few minutes)"):
            for i, review in enumerate(todo):
                result = score_review(review)
                if result is None:
                    failed += 1
                else:
                    cache[(model_name, review)] = result
                progress.progress((i + 1) / len(todo))
    except Exception as err:
        progress.empty()
        set_status(status_name, "error", "SLM run failed (" + str(err)[:200] + "). Retry, or switch model in the sidebar.")
        return False
    progress.empty()
    elapsed = time.time() - started
    summary = "scored " + str(len(todo) - failed) + "/" + str(len(todo)) + " reviews in " + format(elapsed, ".1f") + "s on " + get_device_label()
    if failed:
        set_status(status_name, "info", summary + ". " + str(failed) + " unreadable JSON output(s); the 1.5B model is more reliable.")
    else:
        set_status(status_name, "success", summary)
    return True


def aspect_table(df, pg_names):
    # Averages the cached SLM aspect scores per PG
    model_name = st.session_state.model_name
    cache = st.session_state.aspect_cache
    rows = []
    for pg in pg_names:
        reviews = df.loc[df["pg_name"] == pg, "review"].tolist()
        results = [cache[(model_name, r)] for r in reviews if (model_name, r) in cache]
        if not results:
            continue
        row = {"pg_name": pg, "reviews_scored": len(results)}
        for aspect in ASPECTS:
            values = [x[aspect] for x in results if x.get(aspect) is not None]
            row[aspect] = float(np.mean(values)) if values else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def semantic_search(query, df, top_k=5):
    # Embeds the query and ranks all reviews by cosine similarity
    from sentence_transformers import util
    model = load_embedder()
    mark_loaded("embedder")
    doc_emb = embed_texts(tuple(df["review"].astype(str).tolist()))
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    sims = util.cos_sim(q_emb, doc_emb)[0].cpu().numpy()
    order = np.argsort(-sims)[:top_k]
    rows = []
    for idx in order:
        row = df.iloc[int(idx)]
        rows.append({
            "pg_name": row["pg_name"], "area": row["area"], "rent": row["rent"],
            "gender": row["gender"], "review": row["review"], "score": float(sims[int(idx)]),
        })
    return rows


def filter_pgs(pg_table, budget, gender_choice, office_choice, food_choice):
    # Rule-based shortlist. Also returns how many PGs were left after each rule
    t = pg_table.copy()
    steps = ["all: " + str(len(t))]
    t = t[t["rent"] <= budget]
    steps.append("budget: " + str(len(t)))
    if gender_choice == "Female":
        t = t[t["gender"].isin(["ladies", "unisex"])]
    elif gender_choice == "Male":
        t = t[t["gender"].isin(["gents", "unisex"])]
    steps.append("gender: " + str(len(t)))
    keywords = [k.lower() for k in OFFICE_TO_PG_AREAS.get(office_choice, [])]
    if keywords:
        mask = [any(k in str(a).lower() for k in keywords) for a in t["area"]]
        t = t[pd.Series(mask, index=t.index, dtype=bool)]
    steps.append("office: " + str(len(t)))
    if food_choice == "Vegetarian":
        t = t[t["veg_friendly"]]
    elif food_choice == "Non-vegetarian":
        t = t[t["nonveg_friendly"]]
    steps.append("food: " + str(len(t)))
    return t, steps


def build_reco_prompt(needs_text, rows, df):
    # Builds the user prompt: the needs plus the data of the shortlisted PGs
    lines = ["User needs: " + needs_text, "", "PG data:"]
    for r in rows:
        at = aspect_table(df, [r["pg_name"]])
        if at.empty:
            aspect_text = "no SLM aspect scores available"
        else:
            first = at.iloc[0]
            parts = []
            for aspect in ASPECTS:
                value = first[aspect]
                if pd.notna(value):
                    parts.append(ASPECT_LABELS[aspect] + " " + format(float(value), ".1f"))
                else:
                    parts.append(ASPECT_LABELS[aspect] + " not mentioned")
            aspect_text = ", ".join(parts)
        veg_text = "yes" if r["veg_friendly"] else "no"
        line = (
            "- " + str(r["pg_name"]) + " | area: " + str(r["area"]) + " | rent: Rs " + str(int(r["rent"]))
            + " per month | for: " + str(r["gender"]) + " | veg food mentioned in reviews: " + veg_text
            + " | overall VADER sentiment: " + format(float(r["vader"]), ".2f") + " (" + str(r["sentiment"]) + ")"
            + " | average aspect scores out of 5: " + aspect_text
        )
        lines.append(line)
    return "\n".join(lines)

# ==============================================================
# 7. UI
# ==============================================================
# Keep results and history between reruns
defaults = {
    "aspect_cache": {},
    "status": {},
    "search_results": None,
    "reco_result": None,
    "search_history": [],
    "reco_history": [],
    "last_aspect_prompt": None,
    "loaded": [],
}
for key_name, default_value in defaults.items():
    if key_name not in st.session_state:
        st.session_state[key_name] = default_value

# ---- Sidebar ----
st.sidebar.markdown("### Config")
st.sidebar.selectbox("Small Language Model", SLM_OPTIONS, index=0, key="model_name")
uploaded_file = st.sidebar.file_uploader(
    "Reviews CSV (optional)", type=["csv"], key="csv_upload",
    help="Columns: pg_name, area, rent, gender, review. Leave empty to use the built-in sample.",
)
with st.sidebar.expander("About", expanded=True):
    st.caption("Classic NLP (rules, word counts, small trained models like VADER) scores text from fixed patterns.")
    st.caption("An SLM understands meaning and follows instructions, so it can read a review and explain a choice.")
runtime_box = st.sidebar.empty()

# ---- Header ----
st.title("PG Finder: Smart PG Review Analyst")
st.write("Compare PGs in Marathahalli, BTM Layout and Electronic City on food, safety, Wi-Fi, cleanliness and owner behaviour, using review text instead of broker claims.")

df, source_text = get_dataframe(uploaded_file)
pg_table = build_pg_table(df)
pg_names = pg_table["pg_name"].tolist()

m1, m2, m3, m4 = st.columns(4)
m1.metric("PGs", len(pg_table))
m2.metric("Reviews", len(df))
m3.metric("Median rent", money(pg_table["rent"].median()))
m4.metric("Positive (VADER)", str(int((pg_table["sentiment"] == "Positive").sum())) + "/" + str(len(pg_table)))
st.caption("source: " + source_text)

tab1, tab2, tab3 = st.tabs(["Compare PGs", "Search reviews", "Find my PG"])

# ---------------- TAB 1: COMPARE ----------------
with tab1:
    st.subheader("PG overview")
    st.dataframe(
        pg_display(pg_table, ["pg_name", "area", "rent", "gender", "reviews", "vader"]),
        hide_index=True,
        column_config=PG_COLUMN_CONFIG,
    )

    # Make sure the stored selection only has PGs that exist in the current data
    current = st.session_state.get("compare_pgs")
    if current is None or any(p not in pg_names for p in current):
        st.session_state["compare_pgs"] = pg_names[:2]
    selected = st.multiselect("PGs to compare (max 3)", pg_names, max_selections=3, key="compare_pgs")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Classic NLP")
        st.caption("VADER lexicon score per review, averaged per PG. Range -1 to +1.")
        try:
            fig_v = px.bar(
                pg_table, x="pg_name", y="vader", color="sentiment",
                color_discrete_map={"Positive": COLOR_POS, "Neutral": COLOR_NEU, "Negative": COLOR_NEG},
                labels={"pg_name": "", "vader": "mean compound", "sentiment": ""},
                range_y=[-1, 1], title="VADER sentiment per PG",
            )
            st.plotly_chart(style_fig(fig_v), key="vader_bar_chart")
        except Exception as err:
            st.error("Could not draw the VADER chart: " + str(err)[:150])

    with col_b:
        st.subheader("SLM")
        st.caption("Per-review aspect scores (1-5) extracted as JSON, averaged per PG.")
        if st.button("Score selected PGs", key="btn_score_compare"):
            if not selected:
                st.warning("Select at least one PG first.")
            else:
                st.session_state.status.pop("compare", None)
                reviews_to_score = df[df["pg_name"].isin(selected)]["review"].tolist()
                run_aspect_scoring(reviews_to_score, "compare")
        show_status("compare")

        table = aspect_table(df, selected)
        if table.empty:
            st.info("No SLM scores for the current selection. Run the scoring above.")
        else:
            try:
                long = table.melt(id_vars="pg_name", value_vars=ASPECTS, var_name="Aspect", value_name="Score")
                long["Score"] = long["Score"].fillna(0)
                long["Aspect"] = long["Aspect"].map(ASPECT_LABELS)
                fig_r = px.line_polar(
                    long, r="Score", theta="Aspect", color="pg_name", line_close=True,
                    range_r=[0, 5], title="Mean aspect score (SLM)",
                    color_discrete_sequence=LINE_COLORS,
                )
                fig_r.update_traces(fill="toself", opacity=0.55)
                st.plotly_chart(style_fig(fig_r), key="radar_chart")
                st.caption("0 on the radar means no review mentioned that aspect.")
            except Exception as err:
                st.error("Could not draw the radar chart: " + str(err)[:150])
            st.dataframe(table, hide_index=True, column_config=PG_COLUMN_CONFIG)
            not_scored = [p for p in selected if p not in table["pg_name"].tolist()]
            if not_scored:
                st.info("not scored yet: " + ", ".join(not_scored))

            # AI summaries per PG (one short line per review)
            st.markdown("**Summaries**")
            model_now = st.session_state.model_name
            for pg in table["pg_name"].tolist():
                with st.expander(pg):
                    for review in df.loc[df["pg_name"] == pg, "review"].tolist():
                        item = st.session_state.aspect_cache.get((model_now, review))
                        if item and item.get("summary"):
                            st.write("- " + item["summary"])
        if st.session_state.last_aspect_prompt:
            show_prompts(st.session_state.last_aspect_prompt[0], st.session_state.last_aspect_prompt[1])

# ---------------- TAB 2: SEARCH ----------------
with tab2:
    st.subheader("Semantic review search")
    st.write("Query is embedded with all-MiniLM-L6-v2 and compared to every review by cosine similarity.")
    query = st.text_input("Query", placeholder="quiet place with good Wi-Fi", key="search_query")
    if st.button("Run search", key="btn_search"):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                started = time.time()
                with st.spinner("Searching... (first run downloads the embedding model)"):
                    found_rows = semantic_search(query.strip(), df, top_k=5)
                st.session_state.search_results = {
                    "query": query.strip(), "rows": found_rows, "seconds": time.time() - started,
                }
                st.session_state.search_history.append(query.strip())
            except Exception as err:
                st.error("Search failed: " + str(err)[:200])

    result_box = st.session_state.search_results
    if result_box:
        st.caption("query=" + repr(result_box["query"]) + "  top_k=5  n=" + str(len(df)) + "  time=" + format(result_box["seconds"], ".2f") + "s")
        for rank, item in enumerate(result_box["rows"], start=1):
            with st.container(border=True):
                st.markdown("**" + str(rank) + ". " + str(item["pg_name"]) + "**")
                st.caption(
                    "sim=" + format(item["score"], ".3f") + "  |  " + str(item["area"]) + "  |  "
                    + money(item["rent"]) + "  |  " + str(item["gender"]).title()
                )
                st.write(item["review"])
    else:
        st.info("No results yet. Enter a query and run the search.")
    if st.session_state.search_history:
        with st.expander("Query history"):
            for past in reversed(st.session_state.search_history[-10:]):
                st.code(past, language=None)

# ---------------- TAB 3: RECOMMENDATION ----------------
with tab3:
    st.subheader("Recommender")
    st.write("Rules filter the PGs first, then the SLM picks one PG plus a backup from the shortlist.")
    f1, f2 = st.columns(2)
    with f1:
        budget = st.number_input("Max rent (INR / month)", min_value=3000, max_value=50000, value=10000, step=500, key="reco_budget")
        gender_choice = st.selectbox("Gender", ["Female", "Male", "Other / no preference"], index=0, key="reco_gender")
    with f2:
        office_choice = st.selectbox("Office location", list(OFFICE_TO_PG_AREAS.keys()), index=0, key="reco_office")
        food_choice = st.selectbox("Food", ["Vegetarian", "Non-vegetarian", "No preference"], index=0, key="reco_food")

    if st.button("Run recommendation", key="btn_reco"):
        st.session_state.status.pop("reco", None)
        needs_text = (
            "budget up to Rs " + str(int(budget)) + " per month; gender: " + gender_choice
            + "; office area: " + office_choice + "; food preference: " + food_choice
        )
        shortlist, steps = filter_pgs(pg_table, budget, gender_choice, office_choice, food_choice)
        if shortlist.empty:
            st.session_state.reco_result = {"needs": needs_text, "empty": True, "steps": steps}
        else:
            ranked = shortlist.sort_values("vader", ascending=False).reset_index(drop=True)
            result = {
                "needs": needs_text, "empty": False, "steps": steps,
                "ranked": ranked.to_dict("records"), "slm_text": "",
                "system": RECO_SYSTEM_PROMPT, "user": "", "capped": len(ranked) > 5,
            }
            top_rows = ranked.head(5)
            top_reviews = df[df["pg_name"].isin(top_rows["pg_name"].tolist())]["review"].tolist()
            ok = run_aspect_scoring(top_reviews, "reco")
            if ok:
                user_prompt = build_reco_prompt(needs_text, top_rows.to_dict("records"), df)
                result["user"] = user_prompt
                try:
                    with st.spinner("Thinking with the SLM... (first run downloads the model, so it can take a few minutes)"):
                        result["slm_text"] = slm_chat(RECO_SYSTEM_PROMPT, user_prompt, 300)
                    if not result["slm_text"]:
                        set_status("reco", "info", "The SLM returned an empty answer. Retry, or use the 1.5B model.")
                except Exception as err:
                    set_status("reco", "error", "SLM could not write a recommendation (" + str(err)[:200] + ").")
            st.session_state.reco_result = result
            st.session_state.reco_history.append(needs_text + " -> sentiment pick: " + str(ranked.iloc[0]["pg_name"]))

    res = st.session_state.reco_result
    if res is None:
        st.info("No run yet. Set the filters and press the button.")
    else:
        st.caption("input: " + res["needs"])
        st.code("filter funnel: " + " > ".join(res["steps"]), language=None)
        show_status("reco")
        if res["empty"]:
            st.warning("No PG matches every rule. Raise the budget, change the office area, or relax the food filter.")
        else:
            if res.get("capped"):
                st.info("More than 5 PGs matched; the SLM only saw the 5 with the best VADER sentiment.")
            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("Classic NLP")
                ranked_df = pd.DataFrame(res["ranked"])
                st.caption("Shortlist sorted by mean VADER sentiment.")
                st.dataframe(
                    pg_display(ranked_df, ["pg_name", "area", "rent", "gender", "vader", "sentiment"]),
                    hide_index=True,
                    column_config=PG_COLUMN_CONFIG,
                )
                best = ranked_df.iloc[0]["pg_name"]
                text = "**Top:** " + str(best)
                if len(ranked_df) > 1:
                    text = text + "  \n**Backup:** " + str(ranked_df.iloc[1]["pg_name"])
                st.markdown(text)
            with c_right:
                st.subheader("SLM")
                if res["slm_text"]:
                    st.markdown(res["slm_text"])
                else:
                    st.info("The SLM answer could not be read. Retry, or pick the 1.5B model in the sidebar.")
                if res["user"]:
                    show_prompts(res["system"], res["user"])
    if st.session_state.reco_history:
        with st.expander("Run history"):
            for past in reversed(st.session_state.reco_history[-10:]):
                st.code(past, language=None)

# ---- Sidebar runtime panel (filled last so it always shows the latest state) ----
slm_loaded = "yes" if st.session_state.model_name in st.session_state.loaded else "no"
embed_loaded = "yes" if "embedder" in st.session_state.loaded else "no"
runtime_lines = [
    "device   : " + get_device_label(),
    "slm      : " + st.session_state.model_name.split("/")[-1],
    "slm ready: " + slm_loaded,
    "embedder : " + embed_loaded,
    "scored   : " + str(len(st.session_state.aspect_cache)) + " reviews cached",
]
runtime_box.code("\n".join(runtime_lines), language=None)
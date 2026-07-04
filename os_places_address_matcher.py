# address_matcher_studio_customer.py
#
# Customer-facing edition: one-file address matching tool. Import data
# from CSV or Excel; map one or more address columns plus whatever
# supporting columns you have; match against the OS Places API; download
# the results. No internal infrastructure (NGD PostgreSQL, database
# hosts/credentials) is present in this edition.
#
# Deliberately a single file rather than a package - fewer things to get
# out of sync between files, easier to save/share/rebuild.
#
# Run with: streamlit run address_matcher_studio_customer.py

import importlib.util
import math
import os
import re
import sqlite3
import time
from pathlib import Path

# matplotlib (~1s to import) is only actually needed once results exist
# (the chart) - since Streamlit re-executes this whole file on every
# page load and every widget interaction, importing it unconditionally
# at the top meant paying that cost before the user had even loaded a
# file, on every single rerun. find_spec() checks availability without
# actually loading the module; the real `import matplotlib.pyplot` line
# is deferred to the exact point the chart is drawn (see Section 4).
_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
plt = None  # populated by a lazy import the first time a chart is drawn

import numpy as np
import pandas as pd
import requests
import streamlit as st

from rapidfuzz import fuzz as rfuzz
from rapidfuzz import process as rprocess


def check_password():
    """Simple session-based password gate. Requires a `.streamlit/secrets.toml`
    file (never committed to git - add it to .gitignore) containing:
        APP_PASSWORD = "your-password-here"
    On Streamlit Community Cloud or similar, set the equivalent secret in
    the platform's own settings instead of a local file."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    st.title("OS Places Address Matcher")
    st.info("Authorised Users Only")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


if not check_password():
    st.stop()

# =============================================================================
# SECTION 1: CORE MATCHING ENGINE
# Operates purely on a "candidates" DataFrame with columns [uprn,
# postcode, organisationname, name, fulladdress, easting, northing],
# as returned by the OS Places API.
# =============================================================================

OS_PLACES_BASE = "https://api.os.uk/search/places/v1"

DISTANCE_OUTLIER_THRESHOLD_M = 500
LOW_ADDRESS_SIMILARITY_THRESHOLD = 50
CONFIDENCE_GAP_AMBIGUOUS_THRESHOLD = 5
NAME_CHANGE_SIMILARITY_THRESHOLD = 60
SPATIAL_FALLBACK_RADIUS_M = 250
DUPLICATE_TOP_N = 5
CANDIDATE_DENSITY_LOW_MAX = 5
CANDIDATE_DENSITY_MEDIUM_MAX = 20

REVIEW_STATUS_ORDER = [
    "A EXACT_SITE_MATCH", "B REVIEW_REQUIRED", "C PROBABLE_MATCH", "D NO_MATCH",
]

DEFAULT_NAME_REPLACEMENTS = [("&", "AND"), ("'", "")]
NUMBER_RE = re.compile(r"\b(\d+[A-Za-z]?)\b")
CACHE_COLUMNS = ["postcode_clean", "uprn", "postcode", "organisationname",
                  "name", "fulladdress", "easting", "northing"]


def normalize_name(name, replacements=None):
    if name is None:
        return ""
    n = str(name).upper().strip()
    for old, new in (replacements if replacements is not None else DEFAULT_NAME_REPLACEMENTS):
        n = n.replace(old, new)
    return re.sub(r"\s+", " ", n).strip()


def extract_leading_number(address):
    if not address or pd.isna(address):
        return None
    match = NUMBER_RE.search(str(address))
    return match.group(1).upper() if match else None


def coordinate_band(dist):
    if dist is None or (isinstance(dist, float) and math.isnan(dist)):
        return "UNKNOWN"
    if dist <= 10:
        return "EXACT"
    if dist <= 50:
        return "VERY_CLOSE"
    if dist <= 100:
        return "CLOSE"
    if dist <= 500:
        return "FAR"
    return "OUTLIER"


def candidate_density_band(n):
    if n < CANDIDATE_DENSITY_LOW_MAX:
        return "LOW"
    if n <= CANDIDATE_DENSITY_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def clean_postcode(v):
    if pd.isna(v):
        return ""
    return str(v).replace(" ", "").upper().strip()


def sanitize_filename_stem(name):
    """Strip a file extension and replace anything that isn't safe in a
    Windows filename (< > : " / \\ | ? * and control characters) with an
    underscore, so an output filename built from the source file's name
    can't accidentally fail to save because of a character the source
    filename happened to contain."""
    stem = re.sub(r"\.(csv|xlsx)$", "", name, flags=re.IGNORECASE)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
    return stem.strip() or "matched_addresses"


def postcode_area(postcode_clean):
    """Extract the postcode AREA - the leading letters only (e.g. 'CF'
    from 'CF10 3RB', 'SW' from 'SW1A 1AA', 'B' from 'B2 4QA'). Used to
    sanity-check free-text search fallback results: without any
    coordinates to judge distance, a nationwide text search for a common
    street name (e.g. 'Castle Street') can return a same-named street in
    a completely different part of the country. This is deliberately a
    broad check (area only, not the full outward code) - strict enough to
    catch a match landing in an entirely different region, loose enough
    not to reject a genuinely correct match in a neighbouring district of
    the same town."""
    if not postcode_clean:
        return None
    match = re.match(r"^[A-Z]+", postcode_clean)
    return match.group(0) if match else None


def postcode_outward(postcode_clean):
    """Extract the full outward code (e.g. 'CF10' from 'CF10 3RB', 'BS39'
    from 'BS39 4AE'). Stricter than postcode_area() - same region but a
    different outward code (e.g. Bristol BS1 vs a village at BS39) is a
    real geographic difference worth flagging, even though it wouldn't
    be caught by the area-only check. Used only as an informational flag
    on the output (not a hard filter or grade change), since a
    neighbouring-district match can still be genuinely correct - this
    just makes that situation visible for manual review rather than
    silently indistinguishable from an exact-district match."""
    if not postcode_clean:
        return None
    # Outward code is everything except the last 3 characters (the inward
    # code is always digit+letter+letter, e.g. "3RB", "1AA").
    return postcode_clean[:-3] if len(postcode_clean) > 3 else postcode_clean


# Standard UK postcode pattern (covers all valid formats, including the
# single GIR 0AA special case).
_POSTCODE_RE = re.compile(
    r"([Gg][Ii][Rr]\s?0[Aa]{2})|"
    r"((([A-Za-z][0-9]{1,2})|(([A-Za-z][A-Ha-hJ-Yj-y][0-9]{1,2})|"
    r"(([A-Za-z][0-9][A-Za-z])|([A-Za-z][A-Ha-hJ-Yj-y][0-9]?[A-Za-z]))))"
    r"\s?[0-9][A-Za-z]{2})"
)


def extract_postcode_from_text(text):
    """Pull a UK postcode out of free-form address text, for when no
    dedicated postcode column is mapped but the postcode is embedded in
    the address itself (e.g. "299 Westferry Road, Millwall, London, E14
    3RS" as a single field). Without this, rows with no explicit postcode
    column skip the cheap primary postcode lookup entirely, even when the
    postcode was there all along - going straight to the more expensive,
    less geographically constrained fallback tiers instead."""
    if not text or pd.isna(text):
        return ""
    match = _POSTCODE_RE.search(str(text))
    return match.group(0) if match else ""


_ADDRESS_ABBREV_PATTERNS = [
    (re.compile(r"\bRd\.?\b", re.IGNORECASE), "Road"),
    (re.compile(r"\bAve\.?\b", re.IGNORECASE), "Avenue"),
    (re.compile(r"\bLn\.?\b", re.IGNORECASE), "Lane"),
    (re.compile(r"\bCir\.?\b", re.IGNORECASE), "Circus"),
    (re.compile(r"\bCres\.?\b", re.IGNORECASE), "Crescent"),
    (re.compile(r"\bGdns\.?\b", re.IGNORECASE), "Gardens"),
    (re.compile(r"\bPl\.?\b", re.IGNORECASE), "Place"),
    (re.compile(r"\bSq\.?\b", re.IGNORECASE), "Square"),
    (re.compile(r"\bDr\.?\b", re.IGNORECASE), "Drive"),
    (re.compile(r"\bTer\.?\b", re.IGNORECASE), "Terrace"),
]
# "St" is genuinely ambiguous in UK addresses - "St Martins" means Saint,
# "Priory St" means Street. Only expand it when it's in TRAILING position
# (immediately before a comma or end of string), which is the pattern for
# "Street" as a suffix; leading "St" (followed by another word) is left
# untouched, since that's almost always "Saint" - and expanding it wrongly
# would actively hurt matching against OS's own data, which itself uses
# "ST." as an abbreviation for "Saint" in many records.
_TRAILING_ST_RE = re.compile(r"\bSt\.?(?=\s*(,|$))", re.IGNORECASE)


def expand_address_abbreviations(text):
    """Expand common UK street-type abbreviations in address text before
    fuzzy comparison (Rd->Road, Ave->Avenue, etc.). Only applied to the
    SOURCE address, not candidate addresses - OS Places/NGD data is
    already in full, unabbreviated form, so there's nothing to expand on
    that side."""
    if not text:
        return text
    s = str(text)
    for pattern, replacement in _ADDRESS_ABBREV_PATTERNS:
        s = pattern.sub(replacement, s)
    s = _TRAILING_ST_RE.sub("Street", s)
    return s


def compute_candidate_centroid(candidates_df):
    """Average easting/northing across a postcode's candidate set, as a
    stand-in reference point for distance scoring when the SOURCE row has
    no real coordinates of its own. This only makes sense for candidates
    that already share a genuine postcode (they're geographically
    clustered, typically within a few hundred metres of each other) - it
    would be meaningless for a scattered nationwide free-text search
    result set, so this is only ever used with the primary postcode-based
    candidate pool, not the augmented/fallback one.

    Returns (None, None) if there's nothing usable to average."""
    if candidates_df is None or candidates_df.empty:
        return None, None
    e = pd.to_numeric(candidates_df["easting"], errors="coerce").dropna()
    n = pd.to_numeric(candidates_df["northing"], errors="coerce").dropna()
    if e.empty or n.empty:
        return None, None
    return float(e.mean()), float(n.mean())


def build_combined_address(df, address_cols):
    """Combine one or more address columns into a single string per row,
    skipping blank/missing fields."""
    if not address_cols:
        raise ValueError("At least one address column must be selected")
    return (
        df[address_cols].apply(
            lambda row: ", ".join(str(v).strip() for v in row if pd.notna(v) and str(v).strip()),
            axis=1,
        )
    )


def apply_discount_flags(results_df, discount_col, discount_values_raw):
    """Flag rows based on a source column value (e.g. `EstablishmentStatus
    (name)` = "Closed") without excluding them from the results. This is
    for records that exist and can still be matched perfectly well, but
    are lower-priority for manual review because of something the source
    data itself already tells you (closed, proposed to close, etc.) -
    the flag is informational (a "discounted" attribute plus the reason),
    not a filter, so nothing is silently dropped from the output."""
    results_df = results_df.copy()
    if not discount_col or discount_col not in results_df.columns or not (discount_values_raw or "").strip():
        results_df["discounted"] = False
        results_df["discount_reason"] = None
        return results_df
    values = [v.strip().lower() for v in discount_values_raw.split(",") if v.strip()]
    src_vals = results_df[discount_col].astype(str).str.strip().str.lower()
    mask = src_vals.isin(values)
    results_df["discounted"] = mask
    results_df["discount_reason"] = np.where(
        mask, discount_col + " = " + results_df[discount_col].astype(str), None
    )
    return results_df


def build_comments_column(results_df):
    """Consolidate several independent informational flags - discount
    reason, possible name change, duplicate UPRN conflict, possible
    district mismatch, and distance warning - into one human-readable
    'comments' column for the essential results view, instead of a
    separate boolean/reason column for each. Every underlying flag
    column is untouched and still present in the Full Diagnostic
    download for anyone who wants to filter or sort on them
    individually - this is purely a legibility aid for the everyday
    view, not a replacement for the detail."""
    def _row_comment(row):
        parts = []
        if row.get("discounted"):
            reason = row.get("discount_reason")
            parts.append(f"Discounted ({reason})" if reason else "Discounted")
        if row.get("possible_name_change"):
            parts.append("Possible name change - matched name differs from source")
        if row.get("duplicate_uprn_conflict"):
            parts.append("Duplicate UPRN shared with another row")
        if row.get("possible_district_mismatch"):
            parts.append("Matched postcode district differs from source")
        if row.get("distance_warning"):
            parts.append("UPRN matched but distance exceeds outlier threshold")
        return "; ".join(parts)
    return results_df.apply(_row_comment, axis=1)


def flag_possible_name_change(name_similarity, name_provided, uprn_match, distance_m, address_similarity,
                               threshold=None):
    """Informational only - does NOT affect grade or confidence. A UPRN
    or address match can be entirely correct (the physical location is
    right) while the name at that location has changed since the source
    UPRN was assigned - the site closed and was succeeded by a different
    organisation, the building was repurposed, etc. That is exactly the
    kind of change a stable UPRN is supposed to survive, so a name
    mismatch here is not treated as a matching error and never changes
    the grade - but it's useful to surface, especially on a still-'Open'
    source record, where a poor name match at an otherwise-confirmed
    address could equally mean the source data itself needs a look.

    Deliberately requires the ADDRESS side to be essentially certain
    (UPRN match, near-zero distance, or very high address similarity)
    before flagging on name alone - without this gate, the flag also
    fires on rows where the address itself is uncertain, where a weak
    name match isn't a distinct "possible rename" signal, it's just one
    symptom of an already-ambiguous match that other reasons
    (MULTIPLE_STRONG_CANDIDATES, ZERO_NAME_EVIDENCE, low confidence) are
    already surfacing - flagging those again here would just be noise."""
    threshold = NAME_CHANGE_SIMILARITY_THRESHOLD if threshold is None else threshold
    if not name_provided or name_similarity is None:
        return False
    if name_similarity >= threshold:
        return False
    address_confirmed = bool(uprn_match) or (
        distance_m is not None and not (isinstance(distance_m, float) and math.isnan(distance_m)) and distance_m <= 10
    ) or (address_similarity is not None and address_similarity >= 90)
    return address_confirmed


def normalize_uprn(value):
    """Normalize a UPRN into a plain digit string for comparison, robust
    to common spreadsheet formatting artifacts that a naive str()+strip()
    doesn't catch:
      - thousand-separator commas, e.g. "6,057,967" (a normal Excel
        display format for a plain numeric cell)
      - scientific notation, e.g. "6.057967E+06" - Excel's default
        display for large numbers in a cell that isn't explicitly
        formatted as text, which UPRNs very often aren't
      - a trailing ".0" from a column pandas inferred as float64
      - surrounding whitespace
    Without this, a UPRN that displays correctly in Excel can still fail
    to match a candidate with the identical underlying value, because the
    text representation reaching the app isn't the plain digit string it
    looks like on screen."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip().replace(",", "")
    if not s:
        return ""
    try:
        f = float(s)
        if f == f:  # not NaN
            return str(int(round(f)))
    except (TypeError, ValueError):
        pass
    return s[:-2] if s.endswith(".0") else s


def score_candidates(source_address, source_name, source_easting, source_northing,
                      source_uprn, candidates_df, spatial_fallback=False, name_replacements=None,
                      source_postcode=None):
    if candidates_df is None or len(candidates_df) == 0:
        return None, []

    source_addr = expand_address_abbreviations(source_address or "")
    source_name_norm = normalize_name(source_name or "", name_replacements)

    try:
        se, sn = float(source_easting), float(source_northing)
    except (TypeError, ValueError):
        se = sn = None

    addresses = candidates_df["fulladdress"].fillna("").astype(str).tolist()
    org_names = candidates_df["organisationname"].fillna("").astype(str).tolist()
    site_names = candidates_df["name"].fillna("").astype(str).tolist()
    combined_names = [f"{o} {s}".strip() for o, s in zip(org_names, site_names)]

    combined_norm = [normalize_name(c, name_replacements) for c in combined_names]
    org_norm = [normalize_name(o, name_replacements) for o in org_names]
    site_norm = [normalize_name(s, name_replacements) for s in site_names]

    # Case-insensitive: OS Places/NGD data is typically ALL CAPS, source
    # data typically isn't - without this, an otherwise-perfect match
    # scores far lower purely on casing.
    addr_sim_arr = np.array(
        rprocess.cdist([source_addr.upper()], [a.upper() for a in addresses], scorer=rfuzz.ratio)[0]
    )

    name_scores = []
    for variant in (combined_norm, org_norm, site_norm):
        set_scores = rprocess.cdist([source_name_norm], variant, scorer=rfuzz.token_set_ratio)[0]
        sort_scores = rprocess.cdist([source_name_norm], variant, scorer=rfuzz.token_sort_ratio)[0]
        name_scores.append(np.maximum(np.array(set_scores), np.array(sort_scores)))
    school_sim_arr = np.maximum.reduce(name_scores)

    e_arr = candidates_df["easting"].astype(float).values
    n_arr = candidates_df["northing"].astype(float).values
    if se is not None and sn is not None:
        dist_arr = np.sqrt((e_arr - se) ** 2 + (n_arr - sn) ** 2)
        delta_e_arr, delta_n_arr = np.abs(e_arr - se), np.abs(n_arr - sn)
    else:
        dist_arr = np.full(len(candidates_df), np.nan)
        delta_e_arr = delta_n_arr = np.full(len(candidates_df), np.nan)

    dist_bonus = np.zeros(len(candidates_df))
    dist_bonus = np.where(dist_arr <= 10, 25, dist_bonus)
    dist_bonus = np.where((dist_arr > 10) & (dist_arr <= 25), 20, dist_bonus)
    dist_bonus = np.where((dist_arr > 25) & (dist_arr <= 50), 15, dist_bonus)
    dist_bonus = np.where((dist_arr > 50) & (dist_arr <= 100), 10, dist_bonus)
    dist_bonus = np.where((dist_arr > 100) & (dist_arr <= 250), 5, dist_bonus)
    dist_bonus = np.where(np.isnan(dist_arr), 0, dist_bonus)

    source_number = extract_leading_number(source_addr)
    number_adj = np.zeros(len(candidates_df))
    if source_number is not None:
        for i, addr in enumerate(addresses):
            cand_number = extract_leading_number(addr)
            if cand_number is not None:
                number_adj[i] = 5 if cand_number == source_number else -5

    # Postcode-agreement bonus/penalty. Distance bonus above is often
    # based on a PROXY centroid rather than a real source coordinate
    # (see compute_candidate_centroid) - that proxy can put a candidate
    # in a genuinely different postcode ahead of the correct one purely
    # because it happens to sit a little closer to the centroid. An
    # exact postcode match is a much stronger, ground-truth signal than
    # a distance estimate built from a guessed reference point, so it's
    # rewarded explicitly here rather than relying on distance alone to
    # get it right.
    postcode_adj = np.zeros(len(candidates_df))
    source_postcode_clean = clean_postcode(source_postcode) if source_postcode else ""
    if source_postcode_clean:
        cand_postcodes_clean = candidates_df["postcode"].apply(clean_postcode).values
        exact_pc_match = cand_postcodes_clean == source_postcode_clean
        postcode_adj = np.where(exact_pc_match, 10, postcode_adj)
        # Only penalise a same-outward/different-postcode candidate when
        # we're relying on the unreliable proxy centroid for distance -
        # if the source has real coordinates, the distance signal is
        # trustworthy enough on its own and shouldn't be second-guessed.
        if se is None and sn is None:
            source_outward = postcode_outward(source_postcode_clean)
            cand_outward = np.array([postcode_outward(pc) for pc in cand_postcodes_clean])
            same_outward_diff_postcode = (
                (cand_outward == source_outward) & (~exact_pc_match) & (cand_postcodes_clean != "")
            )
            postcode_adj = np.where(same_outward_diff_postcode, postcode_adj - 5, postcode_adj)

    conf_arr = addr_sim_arr + 10 + dist_bonus + np.minimum(20, school_sim_arr / 5) + number_adj + postcode_adj

    source_uprn_str = normalize_uprn(source_uprn)
    ngd_uprn_arr = candidates_df["uprn"].apply(normalize_uprn).values
    uprn_match_arr = (source_uprn_str != "") & (ngd_uprn_arr == source_uprn_str)
    conf_arr = np.where(uprn_match_arr, np.maximum(conf_arr, 150), conf_arr)

    order = np.argsort(-conf_arr)
    top_n = order[:DUPLICATE_TOP_N]
    candidate_count = len(candidates_df)
    density_band = candidate_density_band(candidate_count)
    method_suffix = "_SPATIAL_FALLBACK" if spatial_fallback else ""

    def build_record(i):
        raw_dist = dist_arr[i]
        dist_val = None if np.isnan(raw_dist) else round(float(raw_dist), 2)
        raw_de, raw_dn = delta_e_arr[i], delta_n_arr[i]
        return {
            "matched_uprn": candidates_df.iloc[i]["uprn"],
            "matched_name": combined_names[i],
            # Raw (uncombined) name fields, kept alongside the combined
            # one - useful for spot-checking exactly what was compared
            # against the source name, rather than only seeing the
            # already-merged result.
            "matched_organisationname_raw": org_names[i],
            "matched_sitename_raw": site_names[i],
            "matched_address": candidates_df.iloc[i]["fulladdress"],
            "matched_postcode": candidates_df.iloc[i]["postcode"],
            "matched_easting": candidates_df.iloc[i]["easting"],
            "matched_northing": candidates_df.iloc[i]["northing"],
            "delta_easting": None if np.isnan(raw_de) else round(float(raw_de), 2),
            "delta_northing": None if np.isnan(raw_dn) else round(float(raw_dn), 2),
            "distance_m": dist_val,
            "coordinate_band": coordinate_band(dist_val),
            "address_similarity": round(float(addr_sim_arr[i]), 2),
            "name_similarity": round(float(school_sim_arr[i]), 1),
            "uprn_match": bool(uprn_match_arr[i]),
            "confidence": round(float(conf_arr[i]), 1),
            "match_method": ("UPRN_EXACT" if uprn_match_arr[i] else "FUZZY_ADDRESS") + method_suffix,
            "candidate_count": candidate_count,
            "candidate_density": density_band,
        }

    top_candidates = [build_record(i) for i in top_n]
    best = dict(top_candidates[0])
    if len(top_candidates) > 1:
        runner_up = top_candidates[1]
        best["runner_up_confidence"] = runner_up["confidence"]
        best["confidence_gap"] = round(best["confidence"] - runner_up["confidence"], 1)
        # Runner-up's actual match detail, not just its score - lets a
        # reviewer see WHAT the second-best option was, not just how
        # close it was, without having to re-run the match themselves.
        best["runner_up_matched_address"] = runner_up["matched_address"]
        best["runner_up_matched_uprn"] = runner_up["matched_uprn"]
    else:
        best["runner_up_confidence"] = None
        best["confidence_gap"] = None
        best["runner_up_matched_address"] = None
        best["runner_up_matched_uprn"] = None
    return best, top_candidates


def grade_match(best, grade_a_threshold=140, grade_c_threshold=95, grade_b_threshold=80,
                 distance_outlier_threshold_m=None, low_address_similarity_threshold=None,
                 confidence_gap_ambiguous_threshold=None, zero_name_evidence_name_provided=False):
    """Grading thresholds are parameters, not hardcoded, so a customer or
    dataset that warrants different sensitivity can be tuned via the UI
    without a code change. All default to the original fixed values, so
    behaviour is identical unless someone explicitly adjusts them."""
    distance_outlier_threshold_m = (
        DISTANCE_OUTLIER_THRESHOLD_M if distance_outlier_threshold_m is None else distance_outlier_threshold_m
    )
    low_address_similarity_threshold = (
        LOW_ADDRESS_SIMILARITY_THRESHOLD if low_address_similarity_threshold is None else low_address_similarity_threshold
    )
    confidence_gap_ambiguous_threshold = (
        CONFIDENCE_GAP_AMBIGUOUS_THRESHOLD if confidence_gap_ambiguous_threshold is None else confidence_gap_ambiguous_threshold
    )

    dist_val = best["distance_m"]
    distance_warning = bool(best["uprn_match"] and dist_val is not None and dist_val > distance_outlier_threshold_m)

    # A source name was supplied but the matched record shares literally
    # no textual evidence with it at all (name_similarity == 0). Address
    # text alone can coincidentally score well against an unrelated
    # record (shared street/town words); zero name evidence for a named
    # landmark/site is a distinct red flag that the confidence total
    # alone doesn't reliably capture, so it forces a review regardless of
    # how the rest of the score comes out.
    zero_name_evidence = bool(zero_name_evidence_name_provided and best.get("name_similarity") == 0)

    if best["uprn_match"] and distance_warning:
        grade = "B REVIEW_REQUIRED"
        explanation = (f"UPRN matched exactly, but the matched address is over "
                        f"{distance_outlier_threshold_m}m away - possible stale/reassigned UPRN")
    elif zero_name_evidence and not best["uprn_match"]:
        grade = "B REVIEW_REQUIRED"
        explanation = "Address text matched, but no name/site evidence links it to the source at all"
    elif best["uprn_match"] or best["confidence"] >= grade_a_threshold:
        grade, explanation = "A EXACT_SITE_MATCH", "UPRN identical, or evidence is effectively exact"
    elif best["confidence"] >= grade_c_threshold:
        grade, explanation = "C PROBABLE_MATCH", "Strong to moderate agreement"
    elif best["confidence"] >= grade_b_threshold:
        grade, explanation = "B REVIEW_REQUIRED", "Some supporting evidence - review recommended"
    else:
        grade, explanation = "D NO_MATCH", "Insufficient evidence"

    gap = best.get("confidence_gap")
    if distance_warning:
        reason = "UPRN_MATCH_DISTANCE_WARNING"
    elif zero_name_evidence and not best["uprn_match"]:
        reason = "ZERO_NAME_EVIDENCE"
    elif best["uprn_match"]:
        reason = "UPRN_MATCH"
    elif dist_val is not None and dist_val > distance_outlier_threshold_m:
        reason = "DISTANCE_OVER_500M"
    elif best["address_similarity"] < low_address_similarity_threshold:
        reason = "LOW_ADDRESS_SIMILARITY"
    elif best["confidence"] < grade_b_threshold:
        reason = "LOW_CONFIDENCE"
    elif gap is not None and gap < confidence_gap_ambiguous_threshold:
        reason = "MULTIPLE_STRONG_CANDIDATES"
    else:
        reason = "STANDARD_MATCH"
    return grade, explanation, reason, distance_warning


def resolve_duplicate_uprns(results):
    order = sorted(range(len(results)),
                    key=lambda i: (results[i].get("confidence") is None, -(results[i].get("confidence") or -1)))
    claimed = {}
    for i in order:
        rec = results[i]
        candidates_list = rec.get("top_candidates") or []
        if not candidates_list:
            rec.setdefault("duplicate_uprn_conflict", False)
            continue
        original_uprn = rec.get("matched_uprn")
        chosen, chosen_idx = None, None
        for j, cand in enumerate(candidates_list):
            uprn = cand.get("matched_uprn")
            if uprn is None or (isinstance(uprn, float) and math.isnan(uprn)):
                continue
            if uprn not in claimed:
                chosen, chosen_idx = cand, j
                break
        if chosen is None:
            rec["duplicate_uprn_conflict"] = True
            continue
        rec["duplicate_uprn_conflict"] = False
        if chosen.get("matched_uprn") != original_uprn:
            preserved = {k: rec.get(k) for k in ("review_status", "review_reason", "match_explanation", "top_candidates")}
            rec.update(chosen)
            rec.update(preserved)
            next_idx = chosen_idx + 1
            if next_idx < len(candidates_list):
                new_runner_up = candidates_list[next_idx]["confidence"]
                rec["runner_up_confidence"] = new_runner_up
                rec["confidence_gap"] = round(chosen["confidence"] - new_runner_up, 1)
            else:
                rec["runner_up_confidence"] = rec["confidence_gap"] = None
            rec["review_reason"] = "REASSIGNED_DUPLICATE_UPRN"
        claimed[chosen["matched_uprn"]] = i
    return results


# =============================================================================
# SECTION 1B: RUN HISTORY / AUDIT LOG
# Persists a record of every match run (who ran what, when, against which
# source, with what result) - separate from the postcode cache, and kept
# even if the cache itself is cleared. Also doubles as the basis for
# cumulative API usage tracking, since "new API calls made" is logged
# per-run and can simply be summed.
# =============================================================================

def init_run_history(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS run_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_timestamp TEXT,
        source_filename TEXT,
        matcher_type TEXT,
        row_count INTEGER,
        unique_postcodes INTEGER,
        new_api_calls INTEGER,
        grade_a INTEGER, grade_b INTEGER, grade_c INTEGER, grade_d INTEGER
    )""")
    conn.commit()
    conn.close()


def log_run(db_path, source_filename, matcher_type, row_count, unique_postcodes,
            new_api_calls, grade_counts):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO run_history
           (run_timestamp, source_filename, matcher_type, row_count, unique_postcodes,
            new_api_calls, grade_a, grade_b, grade_c, grade_d)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), source_filename, matcher_type,
            row_count, unique_postcodes, new_api_calls,
            grade_counts.get("A EXACT_SITE_MATCH", 0), grade_counts.get("B REVIEW_REQUIRED", 0),
            grade_counts.get("C PROBABLE_MATCH", 0), grade_counts.get("D NO_MATCH", 0),
        ),
    )
    conn.commit()
    conn.close()


def get_run_history(db_path, limit=20):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM run_history ORDER BY id DESC LIMIT ?", conn, params=(limit,)
    )
    conn.close()
    return df


def get_cumulative_api_calls(db_path):
    conn = sqlite3.connect(db_path)
    result = conn.execute(
        "SELECT COALESCE(SUM(new_api_calls), 0) FROM run_history WHERE matcher_type = 'OS Places API'"
    ).fetchone()
    conn.close()
    return result[0] if result else 0


# =============================================================================
# SECTION 2: OS PLACES API BACKEND
# =============================================================================

def init_cache(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS ngd_cache (
        postcode_clean TEXT, uprn TEXT, postcode TEXT, organisationname TEXT,
        name TEXT, fulladdress TEXT, easting REAL, northing REAL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ngd_cache_postcode ON ngd_cache(postcode_clean)")
    conn.commit()
    return conn


def load_cached_candidates(conn, postcodes):
    if not postcodes:
        return pd.DataFrame(columns=CACHE_COLUMNS), set()
    placeholders = ",".join("?" for _ in postcodes)
    cached = pd.read_sql_query(f"SELECT * FROM ngd_cache WHERE postcode_clean IN ({placeholders})", conn, params=postcodes)
    found = set(cached["postcode_clean"].unique().tolist()) if not cached.empty else set()
    return cached, found


def store_in_cache(conn, df):
    if df is not None and not df.empty:
        df[CACHE_COLUMNS].to_sql("ngd_cache", conn, if_exists="append", index=False)
        conn.commit()


def _dpa_row_to_candidate(dpa, postcode_clean_for_cache):
    building_bits = " ".join(b for b in [dpa.get("SUB_BUILDING_NAME"), dpa.get("BUILDING_NAME")] if b).strip()
    return {
        "postcode_clean": postcode_clean_for_cache, "uprn": dpa.get("UPRN"), "postcode": dpa.get("POSTCODE"),
        "organisationname": dpa.get("ORGANISATION_NAME") or "", "name": building_bits,
        "fulladdress": dpa.get("ADDRESS") or "", "easting": dpa.get("X_COORDINATE"), "northing": dpa.get("Y_COORDINATE"),
    }


def fetch_postcode_candidates(api_key, postcode, session, max_pages=5, errors=None, max_retries=3):
    postcode_clean = clean_postcode(postcode)
    rows, offset, page_size = [], 0, 100
    for _ in range(max_pages):
        retries = 0
        while True:
            try:
                resp = session.get(f"{OS_PLACES_BASE}/postcode",
                                    params={"postcode": postcode, "key": api_key, "output_srs": "EPSG:27700",
                                            "maxresults": page_size, "offset": offset}, timeout=15)
            except requests.RequestException as e:
                if errors is not None:
                    errors.append({"postcode": postcode, "status": "connection_error", "detail": str(e)})
                return pd.DataFrame(rows, columns=CACHE_COLUMNS) if rows else pd.DataFrame(columns=CACHE_COLUMNS)

            # Rate limited - back off and retry a few times rather than
            # immediately giving up. Large runs (hundreds/thousands of
            # unique postcodes) are the case this actually matters for; a
            # small test file is unlikely to ever hit this.
            if resp.status_code == 429 and retries < max_retries:
                wait = 2 ** retries  # 1s, 2s, 4s
                time.sleep(wait)
                retries += 1
                continue
            break

        if resp.status_code != 200:
            if errors is not None:
                errors.append({"postcode": postcode, "status": resp.status_code, "detail": resp.text[:200]})
            break
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        rows.extend(_dpa_row_to_candidate(r["DPA"], postcode_clean) for r in results if r.get("DPA"))
        total = data.get("header", {}).get("totalresults", len(results))
        offset += page_size
        if offset >= total:
            break
    return pd.DataFrame(rows, columns=CACHE_COLUMNS) if rows else pd.DataFrame(columns=CACHE_COLUMNS)


def fetch_radius_candidates(api_key, easting, northing, session, radius=SPATIAL_FALLBACK_RADIUS_M, errors=None):
    try:
        e, n = float(easting), float(northing)
    except (TypeError, ValueError):
        return pd.DataFrame(columns=CACHE_COLUMNS)
    if math.isnan(e) or math.isnan(n):
        return pd.DataFrame(columns=CACHE_COLUMNS)
    try:
        resp = session.get(f"{OS_PLACES_BASE}/radius",
                            params={"point": f"{e},{n}", "radius": min(radius, 1000), "key": api_key,
                                    "output_srs": "EPSG:27700", "maxresults": 100}, timeout=15)
    except requests.RequestException as ex:
        if errors is not None:
            errors.append({"postcode": f"(radius {e},{n})", "status": "connection_error", "detail": str(ex)})
        return pd.DataFrame(columns=CACHE_COLUMNS)
    if resp.status_code != 200:
        if errors is not None:
            errors.append({"postcode": f"(radius {e},{n})", "status": resp.status_code, "detail": resp.text[:200]})
        return pd.DataFrame(columns=CACHE_COLUMNS)
    rows = [_dpa_row_to_candidate(r["DPA"], clean_postcode(r["DPA"].get("POSTCODE")))
            for r in resp.json().get("results", []) if r.get("DPA")]
    return pd.DataFrame(rows, columns=CACHE_COLUMNS) if rows else pd.DataFrame(columns=CACHE_COLUMNS)


def fetch_text_search_candidates(api_key, address_text, name, session, max_results=20, errors=None):
    query_text = f"{name} {address_text}".strip() if name else (address_text or "")
    if not query_text.strip():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    try:
        resp = session.get(f"{OS_PLACES_BASE}/find",
                            params={"query": query_text, "key": api_key, "output_srs": "EPSG:27700",
                                    "maxresults": max_results}, timeout=15)
    except requests.RequestException as ex:
        if errors is not None:
            errors.append({"postcode": f"(text: {query_text[:40]})", "status": "connection_error", "detail": str(ex)})
        return pd.DataFrame(columns=CACHE_COLUMNS)
    if resp.status_code != 200:
        if errors is not None:
            errors.append({"postcode": f"(text: {query_text[:40]})", "status": resp.status_code, "detail": resp.text[:200]})
        return pd.DataFrame(columns=CACHE_COLUMNS)
    rows = [_dpa_row_to_candidate(r["DPA"], clean_postcode(r["DPA"].get("POSTCODE")))
            for r in resp.json().get("results", []) if r.get("DPA")]
    return pd.DataFrame(rows, columns=CACHE_COLUMNS) if rows else pd.DataFrame(columns=CACHE_COLUMNS)


def needs_augmentation(best, confidence_threshold=95):
    return best is None or best.get("confidence", 0) < confidence_threshold


def augment_candidates(existing, api_key, easting, northing, address_text, name, session,
                        source_postcode=None, errors=None):
    frames = [existing] if existing is not None and not existing.empty else []
    got_radius = False
    if pd.notna(easting) and pd.notna(northing):
        radius_df = fetch_radius_candidates(api_key, easting, northing, session, errors=errors)
        if not radius_df.empty:
            frames.append(radius_df)
            got_radius = True
    if not got_radius:
        text_df = fetch_text_search_candidates(api_key, address_text, name, session, errors=errors)
        if not text_df.empty:
            # Free-text search has no geographic constraint at all - it
            # searches the whole country. Without coordinates to judge
            # distance, a common street name (e.g. "Castle Street") can
            # return a same-named street hundreds of miles away, and
            # nothing in the scoring formula would catch that on its own.
            # If we know the source postcode, constrain results to the
            # same broad area - this is what stops e.g. "Cardiff Castle"
            # matching a "Castle Street" in Cardigan (CF vs SA) instead of
            # actual Cardiff addresses.
            expected_area = postcode_area(clean_postcode(source_postcode)) if source_postcode else None
            if expected_area:
                text_df = text_df[
                    text_df["postcode"].apply(lambda p: postcode_area(clean_postcode(p)) == expected_area)
                ]
            if not text_df.empty:
                frames.append(text_df)
    if not frames:
        return existing if existing is not None else pd.DataFrame(columns=CACHE_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(subset=["uprn"], keep="first").reset_index(drop=True)


def run_os_places_matcher(df, address_cols, postcode_col, name_col, uprn_col, easting_col, northing_col,
                           api_key, cache_db_path=None, progress_callback=None, grade_thresholds=None,
                           name_change_threshold=None):
    cache_db_path = cache_db_path or DEFAULT_CACHE_DB_PATH
    grade_thresholds = grade_thresholds or {}
    df = df.copy()
    df["_combined_address"] = build_combined_address(df, address_cols)

    # If no postcode column was mapped, fall back to extracting one from
    # the address text itself - without this, rows with an embedded but
    # unmapped postcode would skip the cheap primary lookup entirely.
    if postcode_col:
        df["_postcode_for_matching"] = df[postcode_col]
    else:
        df["_postcode_for_matching"] = df["_combined_address"].apply(extract_postcode_from_text)

    session = requests.Session()
    cache_conn = init_cache(cache_db_path)
    fetch_errors = []

    postcodes = (df["_postcode_for_matching"].dropna().astype(str).str.replace(" ", "", regex=False)
                 .str.upper().replace("", pd.NA).dropna().unique().tolist())

    cached_df, cached_postcodes = load_cached_candidates(cache_conn, postcodes)
    missing = [pc for pc in postcodes if pc not in cached_postcodes]
    fetched_frames = []
    for pc in missing:
        candidates = fetch_postcode_candidates(api_key, pc, session, errors=fetch_errors)
        if not candidates.empty:
            fetched_frames.append(candidates)
        # Small courtesy delay between calls - cheap insurance against
        # hitting rate limits on a large run (hundreds/thousands of
        # unique postcodes), on top of the 429 retry-with-backoff already
        # inside fetch_postcode_candidates itself.
        time.sleep(0.05)
        # Fail fast on auth errors - a bad/expired key fails identically
        # on every subsequent call, so there's no point grinding through
        # the rest of the postcodes to find that out one at a time.
        if fetch_errors and fetch_errors[-1]["status"] in (401, 403):
            cache_conn.close()
            raise RuntimeError(
                f"OS Places API rejected the request (HTTP {fetch_errors[-1]['status']}) - "
                f"check your API key is correct and enabled for the Places API product. "
                f"Detail: {fetch_errors[-1]['detail']}"
            )
    fetched_df = pd.concat(fetched_frames, ignore_index=True) if fetched_frames else pd.DataFrame(columns=CACHE_COLUMNS)
    store_in_cache(cache_conn, fetched_df)
    cache_conn.close()

    ngd = pd.concat([cached_df, fetched_df], ignore_index=True)
    postcode_groups = {pc: grp for pc, grp in ngd.groupby("postcode_clean")} if not ngd.empty else {}

    results = []
    total = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        if progress_callback:
            progress_callback(i + 1, total)
        postcode = clean_postcode(row.get("_postcode_for_matching"))
        candidates = postcode_groups.get(postcode)
        easting = row.get(easting_col) if easting_col else None
        northing = row.get(northing_col) if northing_col else None
        address_text = row.get("_combined_address")
        name_value = row.get(name_col) if name_col else None

        # If no real source coordinates were given, fall back to the
        # centroid of the primary postcode's own candidates as a rough
        # distance proxy - better than no distance signal at all, but
        # NOT a substitute for real coordinates, so it's flagged in the
        # output rather than silently presented as if it were real.
        distance_proxy_used = False
        if (pd.isna(easting) or easting is None) and (pd.isna(northing) or northing is None):
            proxy_e, proxy_n = compute_candidate_centroid(candidates)
            if proxy_e is not None:
                easting, northing = proxy_e, proxy_n
                distance_proxy_used = True

        best, top_candidates = score_candidates(address_text, name_value, easting, northing,
                                                  row.get(uprn_col) if uprn_col else None, candidates,
                                                  source_postcode=row.get("_postcode_for_matching"))
        if needs_augmentation(best):
            augmented = augment_candidates(candidates, api_key, easting, northing, address_text, name_value,
                                            session, source_postcode=row.get("_postcode_for_matching"),
                                            errors=fetch_errors)
            if augmented is not None and not augmented.empty:
                best2, top2 = score_candidates(address_text, name_value, easting, northing,
                                                row.get(uprn_col) if uprn_col else None, augmented, spatial_fallback=True,
                                                source_postcode=row.get("_postcode_for_matching"))
                if best2 is not None and (best is None or best2["confidence"] > best["confidence"]):
                    best, top_candidates = best2, top2

        if best is None:
            record = {"matched_address": "NOT FOUND", "matched_uprn": None, "confidence": None,
                      "review_status": "D NO_MATCH", "review_reason": "NO_CANDIDATES_ANY_SOURCE",
                      "match_explanation": "No candidates found via postcode, coordinates, or text search",
                      "top_candidates": [], "candidate_count": 0, "candidate_density": "LOW", "distance_warning": False,
                      "possible_district_mismatch": False, "distance_proxy_used": distance_proxy_used,
                      "possible_name_change": False}
        else:
            zero_name_flag = bool(name_value and str(name_value).strip())
            grade, explanation, reason, warning = grade_match(
                best, zero_name_evidence_name_provided=zero_name_flag, **grade_thresholds)
            best.update(review_status=grade, review_reason=reason, match_explanation=explanation,
                        distance_warning=warning, top_candidates=top_candidates)
            # Informational only - does NOT change the grade or get
            # filtered. Same broad postcode area, different specific
            # outward code (e.g. Bristol BS1 vs a village at BS39) can
            # still be a genuinely correct match, so this just makes the
            # difference visible for manual review rather than silently
            # treating it the same as an exact-district match.
            src_outward = postcode_outward(postcode) if postcode else None
            matched_outward = postcode_outward(clean_postcode(best.get("matched_postcode")))
            best["possible_district_mismatch"] = bool(
                src_outward and matched_outward and src_outward != matched_outward
            )
            best["distance_proxy_used"] = distance_proxy_used
            best["possible_name_change"] = flag_possible_name_change(
                best.get("name_similarity"), zero_name_flag, best.get("uprn_match"),
                best.get("distance_m"), best.get("address_similarity"), name_change_threshold
            )
            record = best
        results.append(record)


    results = resolve_duplicate_uprns(results)
    for rec in results:
        rec.pop("top_candidates", None)
        rec.setdefault("duplicate_uprn_conflict", False)
    matches_df = pd.DataFrame(results)
    final_df = pd.concat([df.reset_index(drop=True), matches_df.reset_index(drop=True)], axis=1)
    # Internal working columns - useful during matching, but the address
    # combination and extracted postcode are already implicit in the
    # source columns and the matched_* fields. Dropped here (once, at the
    # source) rather than filtered per-download, so every export -
    # essential, review-only, and Full Diagnostic alike - stays free of
    # them without needing to remember to exclude them in three places.
    final_df = final_df.drop(columns=["_combined_address", "_postcode_for_matching"], errors="ignore")
    return final_df, fetch_errors, len(missing)


# SECTION 4: STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Address Matcher Studio", layout="wide")

# Stable cache location rather than a bare relative filename - a relative
# path means the cache location silently changes depending on which
# folder you happen to run `streamlit run` from, losing the caching
# benefit across sessions started from different places.
_APP_DATA_DIR = Path("C:/temp/address_matcher") if os.name == "nt" else Path.home() / ".address_matcher"
_APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CACHE_DB_PATH = str(_APP_DATA_DIR / "os_places_cache.sqlite")
RUN_HISTORY_DB_PATH = str(_APP_DATA_DIR / "run_history.sqlite")
init_run_history(RUN_HISTORY_DB_PATH)
st.title("📍 Address Matcher Studio")
st.caption("Import from CSV or Excel. Match against the OS Places API.")

if "source_df" not in st.session_state:
    st.session_state["source_df"] = None
if "source_filename" not in st.session_state:
    st.session_state["source_filename"] = "matched_addresses"
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "fetch_errors" not in st.session_state:
    st.session_state["fetch_errors"] = []
if "discount_applied" not in st.session_state:
    st.session_state["discount_applied"] = False

# --- Sidebar ---
st.sidebar.header("OS Places API")
_env_api_key = os.environ.get("OS_PLACES_API_KEY", "")
os_places_api_key = st.sidebar.text_input(
    "API key", type="password", value=_env_api_key,
    help="Pre-filled from the OS_PLACES_API_KEY environment variable if set, otherwise enter manually.",
)
os_places_api_key = os_places_api_key.strip() if os_places_api_key else os_places_api_key

if os_places_api_key:
    # Diagnostic only - never shows the full key. If this doesn't match
    # what you see on the OS Data Hub dashboard (same length, same first/
    # last few characters), the key is being altered somewhere before
    # this point - most commonly a trailing space/newline picked up when
    # copy-pasting, which .strip() above should now catch, but this lets
    # you visually confirm it.
    st.sidebar.caption(
        f"Key received: {len(os_places_api_key)} characters, "
        f"starts '{os_places_api_key[:4]}...', ends '...{os_places_api_key[-4:]}'"
    )
    if _env_api_key:
        st.sidebar.caption("✓ Loaded from OS_PLACES_API_KEY environment variable.")
st.sidebar.caption("Never stored - only held in memory for this session.")

with st.sidebar.expander("📜 Run history"):
    _history_df = get_run_history(RUN_HISTORY_DB_PATH, limit=20)
    if _history_df.empty:
        st.caption("No runs logged yet.")
    else:
        st.dataframe(_history_df, width='stretch', hide_index=True)
        st.caption(
            "Persistent audit trail of every match run on this machine - "
            "survives even if the postcode cache is cleared."
        )

with st.sidebar.expander("Advanced: grading thresholds"):
    st.caption(
        "Adjust how strict each grade tier is. Defaults match the "
        "standard thresholds used throughout testing - only change "
        "these if you have a specific reason to (e.g. a dataset where "
        "the defaults consistently over- or under-grade matches you can "
        "verify by other means)."
    )
    grade_a_threshold = st.number_input("Grade A: confidence ≥", value=140, min_value=80, max_value=300, step=5)
    grade_c_threshold = st.number_input("Grade C: confidence ≥", value=95, min_value=50, max_value=200, step=5)
    grade_b_threshold = st.number_input("Grade B: confidence ≥", value=80, min_value=0, max_value=150, step=5)
    distance_outlier_threshold_m = st.number_input("Distance outlier warning (metres) >", value=500, min_value=50, max_value=5000, step=50)
    low_address_similarity_threshold = st.number_input("Low address similarity <", value=50, min_value=0, max_value=100, step=5)
    confidence_gap_ambiguous_threshold = st.number_input("Ambiguous match: confidence gap <", value=5, min_value=0, max_value=50, step=1)
    name_change_similarity_threshold = st.number_input(
        "Possible name change: name similarity <", value=60, min_value=0, max_value=100, step=5,
        help="Informational only - doesn't affect grade. Flags a row where the site/address is "
             "confirmed but the matched name is a poor match to the source name, e.g. a closed or "
             "renamed site.",
    )

# --- 1. Input source ---
st.header("1. Import data")
uploaded = st.file_uploader(
    "Drag and drop a CSV or Excel file here, or click to browse",
    type=["csv", "xlsx"],
)
if uploaded is not None and st.button("Load File"):
    with st.spinner(f"Loading {uploaded.name}..."):
        try:
            if uploaded.name.lower().endswith(".xlsx"):
                df = pd.read_excel(uploaded, dtype=str)
            else:
                df = pd.read_csv(uploaded, dtype=str)
            df = df.dropna(how="all").reset_index(drop=True)
            st.session_state["source_df"] = df
            st.session_state["source_filename"] = sanitize_filename_stem(uploaded.name)
            st.success(f"{len(df):,} rows loaded from {uploaded.name}")
        except Exception as ex:
            st.error(f"Couldn't read {uploaded.name}: {ex}")

# --- Everything below requires data to be loaded first ---
if st.session_state["source_df"] is not None:
    df = st.session_state["source_df"]

    st.header("2. Preview")
    st.dataframe(df.head(10), width='stretch')

    if len(df.columns) == 1:
        st.warning(
            f"⚠️ Only one column was detected: `{df.columns[0][:60]}...`. "
            "If your data should have several columns (Name, Address, "
            "Postcode, etc.), this usually means the file wasn't split "
            "into columns properly when it was created - commas may have "
            "been pasted as literal text into a single cell/column rather "
            "than being read as column separators. Check the source file "
            "before mapping columns below, or re-upload the original CSV "
            "directly if you have it."
        )

    if len(df) > 5000:
        st.info(
            f"ℹ️ This file has {len(df):,} rows. Large runs can take a while "
            f"(each uncached postcode is a separate API/database call) - "
            f"consider testing with a smaller sample first to confirm your "
            f"column mapping and settings are right before committing to a "
            f"full run."
        )
    columns = list(df.columns)

    st.header("3. Map columns")
    address_cols = st.multiselect(
        "Address column(s) - select one, or several to combine (e.g. Street, Town, Postcode)", columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        name_col = st.selectbox("Name / organisation (optional)", [""] + columns) or None
        uprn_col = st.selectbox("Existing UPRN (optional)", [""] + columns) or None
    with c2:
        postcode_col = st.selectbox("Postcode (optional, recommended)", [""] + columns) or None
    with c3:
        easting_col = st.selectbox("Easting (optional)", [""] + columns) or None
        northing_col = st.selectbox("Northing (optional)", [""] + columns) or None

    if address_cols:
        preview = build_combined_address(df, address_cols)
        st.caption("Combined address preview:")
        st.dataframe(pd.DataFrame({"Combined Address": preview.head(5)}), width='stretch', hide_index=True)

    with st.expander("Optional: discount rule"):
        st.caption(
            "Flag rows based on a source column value - e.g. mark records "
            "where 'EstablishmentStatus (name)' is 'Closed' - without "
            "removing them from the results. Flagged rows get a "
            "'discounted' column plus a reason in the output, and are left "
            "out of the manual-review count below, since they typically "
            "don't need the same scrutiny as an active record."
        )
        discount_col = st.selectbox("Column to check (optional)", [""] + columns) or None
        discount_values_raw = st.text_input(
            "Value(s) to flag as discounted (comma-separated, case-insensitive)",
            value="", disabled=not discount_col,
            help="e.g. Closed, Proposed to close",
        )

    st.header("4. Match")
    matcher_type = "OS Places API"
    ready = len(address_cols) > 0

    if not ready:
        st.info("Select at least one address column above.")

    _unique_postcode_count = 0
    if ready:
        if postcode_col:
            _est_postcodes = df[postcode_col]
        else:
            _est_addr_preview = build_combined_address(df, address_cols)
            _est_postcodes = _est_addr_preview.apply(extract_postcode_from_text)
        _unique_postcode_count = (
            _est_postcodes.dropna().astype(str).str.replace(" ", "", regex=False)
            .str.upper().replace("", pd.NA).dropna().nunique()
        )
        # Estimate postcode lookups before committing to a run - this is
        # the unit OS Places actually bills against, not row count, and
        # it's easy to lose track of that distinction when a file has
        # many rows sharing few postcodes (cheap) vs. mostly unique ones
        # (each an individual paid call).
        st.caption(
            f"📊 Estimated OS Places API usage: up to **{_unique_postcode_count:,}** postcode "
            f"lookups for {len(df):,} rows (postcodes already cached from a previous run don't "
            f"count again; rows sharing a postcode only cost one lookup between them)."
        )
        _cumulative_calls = get_cumulative_api_calls(RUN_HISTORY_DB_PATH)
        if _cumulative_calls:
            st.caption(f"Cumulative OS Places API calls logged so far (all runs, this machine): **{_cumulative_calls:,}**")

    if ready and st.button("Run Matching", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def _progress(cur, tot):
            progress_bar.progress(cur / tot)
            status_text.text(f"Matching row {cur}/{tot}")

        _grade_thresholds = {
            "grade_a_threshold": grade_a_threshold,
            "grade_c_threshold": grade_c_threshold,
            "grade_b_threshold": grade_b_threshold,
            "distance_outlier_threshold_m": distance_outlier_threshold_m,
            "low_address_similarity_threshold": low_address_similarity_threshold,
            "confidence_gap_ambiguous_threshold": confidence_gap_ambiguous_threshold,
        }

        try:
            if not os_places_api_key:
                st.error("Enter an OS Places API key in the sidebar.")
                st.stop()
            results_df, fetch_errors, new_api_calls = run_os_places_matcher(
                df, address_cols, postcode_col, name_col, uprn_col, easting_col, northing_col,
                os_places_api_key, progress_callback=_progress, grade_thresholds=_grade_thresholds,
                name_change_threshold=name_change_similarity_threshold)
            # Persisted to session_state (not just shown inline) so the
            # detail survives page reruns and is downloadable after the
            # fact - previously this only ever existed in this one
            # expander for the duration of the run that produced it.
            st.session_state["fetch_errors"] = fetch_errors
            if fetch_errors:
                with st.expander(f"⚠️ {len(fetch_errors)} API request(s) failed - click for details"):
                    st.dataframe(pd.DataFrame(fetch_errors))
                    st.caption(
                        "If every request failed with a connection error, this usually means "
                        "a network/proxy/SSL issue rather than a bad API key - if you're on a "
                        "corporate network with TLS inspection, check whether pip-system-certs "
                        "is installed and whether it's actually resolving the issue. If requests "
                        "are failing with HTTP 401 or 403, the API key itself is the problem."
                    )

            results_df = apply_discount_flags(results_df, discount_col, discount_values_raw)
            results_df["comments"] = build_comments_column(results_df)
            st.session_state["discount_applied"] = bool(discount_col and (discount_values_raw or "").strip())
            st.session_state["results_df"] = results_df
            status_text.text("Matching complete")
            progress_bar.progress(1.0)
            st.success(f"Matched {len(results_df):,} rows")

            # Log this run to the persistent audit trail - what was run,
            # when, against which source, with what result - independent
            # of the postcode cache, so it survives even if the cache
            # itself is cleared.
            if "review_status" in results_df.columns:
                _grade_counts = results_df["review_status"].value_counts().to_dict()
            else:
                _grade_counts = {}
            log_run(
                RUN_HISTORY_DB_PATH,
                source_filename=st.session_state.get("source_filename", "unknown"),
                matcher_type=matcher_type,
                row_count=len(results_df),
                unique_postcodes=_unique_postcode_count,
                new_api_calls=new_api_calls,
                grade_counts=_grade_counts,
            )
        except Exception as ex:
            st.error(f"Matching failed: {ex}")

    # --- Results ---
    if st.session_state["results_df"] is not None:
        st.header("5. Results")
        results_df = st.session_state["results_df"]

        # Essential columns: source data (whatever the person actually
        # had) plus only the fields that speak directly to match quality.
        # Everything else (raw scoring internals, runner-up detail,
        # coordinates, raw name fields) is still available via the
        # toggle below - nothing is lost, just hidden by default so a
        # routine review isn't scrolling through 30+ columns to find the
        # handful that actually matter.
        # Essential columns: source data plus only the fields needed for
        # a quick pass/fail read. confidence, review_reason,
        # possible_district_mismatch, duplicate_uprn_conflict,
        # discount_reason, and possible_name_change are deliberately left
        # out of this default view (still in the Full Diagnostic download)
        # - review_status plus the colour-coding below already conveys
        # "how good is this match" at a glance, and anything those flags
        # would otherwise add individually is already folded into the
        # single 'comments' column instead. 'discounted' itself is kept
        # as its own visible boolean column (rather than only appearing
        # inside 'comments' as text) so it can be sorted/filtered on
        # directly in the table.
        source_columns = [c for c in df.columns if c not in ("_combined_address",)]
        essential_match_columns = [
            "matched_address", "matched_uprn", "matched_postcode",
            "review_status", "discounted", "match_explanation", "comments",
        ]
        essential_columns = source_columns + [c for c in essential_match_columns if c in results_df.columns]

        if "review_status" in results_df.columns:
            breakdown = results_df["review_status"].value_counts().reindex(REVIEW_STATUS_ORDER, fill_value=0)
            _breakdown_df = breakdown.rename("count").reset_index()
            _breakdown_df.columns = ["review_status", "count"]
            _total = _breakdown_df["count"].sum()
            _breakdown_df["percentage"] = (_breakdown_df["count"] / _total * 100) if _total else 0.0
            _breakdown_df["label"] = _breakdown_df["percentage"].round(1).astype(str) + "%"

            # matplotlib's bar_label instead of Altair/st.bar_chart - a
            # layered Altair chart's text marks are prone to being
            # clipped or invisible depending on the Altair/Streamlit
            # version in use (the tallest bar's label in particular sits
            # right at the plot's clip edge). bar_label is a plain,
            # version-stable way to guarantee the percentage actually
            # renders above every bar. Falls back to the plain built-in
            # chart plus a text caption if matplotlib isn't installed in
            # this environment, so a missing dependency degrades the
            # display rather than crashing the whole app.
            if _HAS_MATPLOTLIB:
                import matplotlib.pyplot as plt  # lazy import - see note at top of file
                _grade_colors = {
                    "A EXACT_SITE_MATCH": "#2ecc71", "B REVIEW_REQUIRED": "#f1c40f",
                    "C PROBABLE_MATCH": "#3498db", "D NO_MATCH": "#e74c3c",
                }
                # Streamlit's default dark theme background (#0e1117) and
                # near-white text, rather than matplotlib's own default
                # white figure - otherwise this chart sits as a bright
                # white rectangle in an otherwise dark app. Also sized
                # down (and dpi bumped) since the original 6x3.2" figure
                # rendered oversized relative to the rest of the page.
                _bg = "#0e1117"
                _fg = "#fafafa"
                _fig, _ax = plt.subplots(figsize=(4.2, 2.4), dpi=150)
                _fig.patch.set_facecolor(_bg)
                _ax.set_facecolor(_bg)
                _bar_colors = [_grade_colors.get(s, "#999999") for s in _breakdown_df["review_status"]]
                _bars = _ax.bar(_breakdown_df["review_status"], _breakdown_df["count"], color=_bar_colors)
                _ax.bar_label(_bars, labels=_breakdown_df["label"], padding=3, color=_fg, fontsize=8)
                _ax.set_ylabel("Rows", color=_fg, fontsize=9)
                _ax.set_ylim(0, max(_breakdown_df["count"].max(), 1) * 1.15)  # headroom so labels aren't clipped
                _ax.tick_params(colors=_fg, labelsize=8)
                for _spine in _ax.spines.values():
                    _spine.set_color(_fg)
                plt.setp(_ax.get_xticklabels(), rotation=15, ha="right", color=_fg)
                _fig.tight_layout()
                st.pyplot(_fig, use_container_width=False)
            else:
                st.bar_chart(_breakdown_df.set_index("review_status")["count"])
                st.caption(
                    "  •  ".join(f"{r.review_status}: {r.label}" for r in _breakdown_df.itertuples())
                    + "  (install matplotlib - `pip install matplotlib` - for percentage labels directly on the bars)"
                )

            # Discounted rows (e.g. closed schools) are excluded from the
            # review count and the "review cases only" export - they're
            # still fully visible in every other download with their
            # discounted/discount_reason flag intact, just deprioritised
            # from the manual-review workload rather than hidden.
            not_discounted = ~results_df.get("discounted", pd.Series(False, index=results_df.index)).fillna(False)
            review_mask = (
                (results_df.get("distance_m", pd.Series(dtype=float)).fillna(0) > DISTANCE_OUTLIER_THRESHOLD_M)
                | (results_df.get("confidence", pd.Series(dtype=float)).fillna(999) < 80)
                | (results_df.get("duplicate_uprn_conflict", False) == True)  # noqa: E712
            ) & not_discounted
            st.write(f"**{review_mask.sum():,} rows flagged for manual review**")
            if st.session_state.get("discount_applied"):
                _discounted_count = int((~not_discounted).sum())
                st.caption(f"{_discounted_count:,} row(s) discounted by the rule above and excluded from this count.")

            # Informational only, deliberately NOT folded into review_mask -
            # a name mismatch at an otherwise-confirmed address/UPRN is not
            # a matching error (that's exactly what a stable UPRN is meant
            # to survive), so it doesn't add to the review workload above,
            # but it's surfaced here since it's easy to miss otherwise -
            # particularly on Grade A / UPRN-matched rows, where nothing
            # else about the row would hint at it.
            _name_change_count = int(results_df.get("possible_name_change", pd.Series(dtype=bool)).fillna(False).sum())
            if _name_change_count:
                st.caption(
                    f"ℹ️ {_name_change_count:,} row(s) have a confirmed address/UPRN but a matched name that's a "
                    f"poor match to the source name - possible closure, rename, or site repurposing since the "
                    f"UPRN was last verified. See the 'possible_name_change' column in the full diagnostic download."
                )

        show_all_columns = st.checkbox(
            "Show all diagnostic columns (scoring internals, coordinates, runner-up detail)",
            value=False,
        )
        display_df = results_df if show_all_columns else results_df[essential_columns]

        # Leading dot indicator instead of full-row shading - same colour
        # meaning as before (green=confirmed, blue=probable,
        # amber=review required, red=no match), just as a compact symbol
        # at the start of each row rather than colouring the whole line.
        _GRADE_DOTS = {
            "A EXACT_SITE_MATCH": "🟢",
            "C PROBABLE_MATCH": "🔵",
            "B REVIEW_REQUIRED": "🟡",
            "D NO_MATCH": "🔴",
        }

        display_head = display_df.head(50).copy()
        if "review_status" in display_head.columns:
            st.caption("🟢 A (confirmed)   🔵 C (probable)   🟡 B (review required)   🔴 D (no match)")
            display_head.insert(0, "Match", display_head["review_status"].map(_GRADE_DOTS).fillna(""))
        st.dataframe(display_head, width='stretch')

        stem = st.session_state.get("source_filename", "matched_addresses")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.download_button("⬇️ Download Results (essential columns)",
                                data=results_df[essential_columns].to_csv(index=False).encode("utf-8"),
                                file_name=f"{stem}_matched_{timestamp}.csv", mime="text/csv")
        with col_b:
            if "review_status" in results_df.columns:
                st.download_button("⬇️ Download Review Cases Only",
                                    data=results_df[review_mask][essential_columns].to_csv(index=False).encode("utf-8"),
                                    file_name=f"{stem}_review_{timestamp}.csv", mime="text/csv")
        with col_c:
            st.download_button("⬇️ Download Full Diagnostic Data",
                                data=results_df.to_csv(index=False).encode("utf-8"),
                                file_name=f"{stem}_full_diagnostic_{timestamp}.csv", mime="text/csv")
        with col_d:
            # Persisted in session_state at run time (see "4. Match" above)
            # so this stays downloadable even after the in-app expander
            # that first showed the failures has been collapsed/scrolled
            # past, or the run that produced them wasn't the most recent
            # rerun of the page.
            _fetch_errors = st.session_state.get("fetch_errors") or []
            if _fetch_errors:
                st.download_button("⬇️ Download Failed Requests",
                                    data=pd.DataFrame(_fetch_errors).to_csv(index=False).encode("utf-8"),
                                    file_name=f"{stem}_failed_requests_{timestamp}.csv", mime="text/csv")
            else:
                st.caption("No failed API requests logged for this run.")


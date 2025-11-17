"""
Text-only longitudinal summarization demo using vLLM.

This script illustrates how to:
  1. Represent multimodal visit bundles as modality-tagged text.
  2. Run a sliding-window summarization pass with an LLM.
  3. Carry forward the previous summary as contextual memory for the next visit.

It ships with a small dummy patient containing three visits (two real, one virtual)
so the pipeline can be exercised without waiting for the full preprocessing output.
Once real v0/v1 artifacts are available, replace the dummy loader with a reader that
emits the same data structure.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Any, Dict, Set

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from icdmappings import Mapper  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Mapper = None  # type: ignore


# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------


DEMO_CANDIDATE_SUBJECTS = [
    12245786,
    18530425,
    16321205,
    19150427,
    15230030,
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ModalityText:
    """Represents a single modality-specific text snippet for a visit."""

    modality: str
    text: str

    def to_prompt_segment(self) -> str:
        header = self.modality.upper()
        return f"{header}:\n{self.text.strip()}"


@dataclass
class VisitNarrative:
    """Holds all modality texts for a single patient visit."""

    visit_id: str
    subject_id: Optional[int] = None
    is_virtual: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    modalities: List[ModalityText] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        tag = "virtual" if self.is_virtual else "real"
        header = f"Visit {self.visit_id} ({tag})"
        if self.start_time or self.end_time:
            header += f" [{self.start_time or '?'} -> {self.end_time or '?'}]"
        modality_blocks = "\n\n".join(mt.to_prompt_segment() for mt in self.modalities)
        return f"{header}\n{modality_blocks}"


# ---------------------------------------------------------------------------
# Real-data helpers
# ---------------------------------------------------------------------------


def _safe_iso(value) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime().isoformat()
        except Exception:
            pass
    return str(value)


def _sort_key_for_record(record: dict) -> tuple:
    raw_time = record.get("admittime")
    dt_obj = None
    if hasattr(raw_time, "to_pydatetime"):
        try:
            dt_obj = raw_time.to_pydatetime()
        except Exception:
            dt_obj = None
    elif isinstance(raw_time, datetime):
        dt_obj = raw_time
    elif raw_time is not None:
        try:
            dt_obj = datetime.fromisoformat(str(raw_time))
        except Exception:
            dt_obj = None
    return (record.get("subject_id"), dt_obj, record.get("hadm_id"))


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


_ICD_CATEGORY_CACHE: Optional[
    Tuple[
        Dict[str, str],
        Dict[str, Set[str]],
        Dict[str, str],
        Dict[str, Set[str]],
    ]
] = None


def _truncate_icd9(code: str, length: int = 3) -> str:
    code = code.strip().upper().replace(".", "")
    if not code:
        return ""
    if code[0] in {"E", "V"}:
        return code[: length + 1]
    return code[:length]


def _truncate_icd10(code: str, length: int = 3) -> str:
    code = code.strip().upper().replace(".", "")
    if not code:
        return ""
    return code[:length]


_ICD_MAPPER: Optional["Mapper"] = None


def _get_icd_mapper() -> Optional["Mapper"]:
    global _ICD_MAPPER
    if Mapper is None:
        return None
    if _ICD_MAPPER is None:
        try:
            _ICD_MAPPER = Mapper()
        except Exception:
            _ICD_MAPPER = None
    return _ICD_MAPPER


def _load_icd_category_maps() -> Tuple[
    Dict[str, str],
    Dict[str, Set[str]],
    Dict[str, str],
    Dict[str, Set[str]],
]:
    """
    Returns cached mappings:
      - ccs_code_to_desc
      - icd9_to_ccs
      - ccsr_code_to_desc
      - icd10_to_ccsr
    Falls back to empty mappings if source files are unavailable.
    """
    global _ICD_CATEGORY_CACHE
    if _ICD_CATEGORY_CACHE is not None:
        return _ICD_CATEGORY_CACHE

    base_dir = Path(__file__).resolve().parent
    ccs_path = base_dir / "AppendixASingleDX.txt"
    ccsr_path = base_dir / "DXCCSR_v2025-1.csv"

    ccs_code_to_desc: Dict[str, str] = {}
    icd9_to_ccs: Dict[str, Set[str]] = defaultdict(set)
    ccsr_code_to_desc: Dict[str, str] = {}
    icd10_to_ccsr: Dict[str, Set[str]] = defaultdict(set)

    if ccs_path.exists():
        current_ccs: Optional[str] = None
        with ccs_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                # CCS code lines start at column 0 (no leading whitespace)
                # ICD code lines start with whitespace
                if raw_line[0] != ' ' and raw_line[0] != '\t':
                    parts = line.split(None, 1)
                if parts and parts[0].isdigit():
                    current_ccs = parts[0]
                    desc = parts[1].strip() if len(parts) > 1 else ""
                    ccs_code_to_desc[current_ccs] = desc
                    continue
                if current_ccs is None:
                    continue
                # This is an ICD code line (starts with whitespace)
                for code in line.split():
                    norm = code.strip().upper().replace(".", "")
                    if norm:
                        icd9_to_ccs[norm].add(current_ccs)
                        trunc = _truncate_icd9(norm)
                        if trunc and trunc != norm:
                            icd9_to_ccs[trunc].add(current_ccs)

    if ccsr_path.exists():
        with ccsr_path.open("r", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                icd10 = row[0].strip().upper().replace(".", "")
                if not icd10:
                    continue
                trunc10 = _truncate_icd10(icd10)
                # Column indices: (code, desc) pairs at positions (2,3), (4,5), ..., (16,17)
                for idx in range(2, min(len(row) - 1, 17), 2):
                    code = row[idx].strip().upper()
                    if not code:
                        continue
                    desc = row[idx + 1].strip() if idx + 1 < len(row) else ""
                    icd10_to_ccsr[icd10].add(code)
                    if trunc10 and trunc10 != icd10:
                        icd10_to_ccsr[trunc10].add(code)
                    if code not in ccsr_code_to_desc and desc:
                        ccsr_code_to_desc[code] = desc

    _ICD_CATEGORY_CACHE = (
        ccs_code_to_desc,
        icd9_to_ccs,
        ccsr_code_to_desc,
        icd10_to_ccsr,
    )
    return _ICD_CATEGORY_CACHE


def _to_iterable(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _render_diagnoses(diagnoses: List[dict]) -> str:
    if not diagnoses:
        return ""
    icd_codes_by_version: Dict[str, List[str]] = defaultdict(list)
    icd_extra_counts: Dict[str, int] = defaultdict(int)
    ccs_codes: Set[str] = set()
    ccsr_codes: Set[str] = set()

    ccs_code_to_desc, icd9_to_ccs, ccsr_code_to_desc, icd10_to_ccsr = _load_icd_category_maps()
    mapper = _get_icd_mapper()

    MAX_CODES_PER_VERSION = 25

    for diag in diagnoses:
        icd_code = str(diag.get("icd_code", "")).strip()
        if not icd_code:
            continue

        version = diag.get("icd_version")
        if version == 9 or version == "9":
            version_label = "ICD-9"
        elif version == 10 or version == "10":
            version_label = "ICD-10"
        else:
            version_label = "ICD"

        codes_list = icd_codes_by_version[version_label]
        if icd_code not in codes_list:
            if len(codes_list) < MAX_CODES_PER_VERSION:
                codes_list.append(icd_code)
            else:
                icd_extra_counts[version_label] += 1

        norm_code = icd_code.replace(".", "").upper()

        if version_label == "ICD-9":
            mapped_ccs = set(icd9_to_ccs.get(norm_code, []))
            if mapper is not None:
                mapped_ccs.update(str(c) for c in _to_iterable(mapper.map(icd_code, source="icd9", target="ccs")))
            ccs_codes.update(code for code in mapped_ccs if code in ccs_code_to_desc)
        elif version_label == "ICD-10":
            mapped_ccsr = set(icd10_to_ccsr.get(norm_code, []))
            if mapper is not None:
                mapped_ccsr.update(str(c) for c in _to_iterable(mapper.map(icd_code, source="icd10", target="ccsr")))
            ccsr_codes.update(code for code in mapped_ccsr if code in ccsr_code_to_desc)
        else:
            if mapper is not None:
                mapped_ccs = set(str(c) for c in _to_iterable(mapper.map(icd_code, source="icd9", target="ccs")))
                mapped_ccsr = set(str(c) for c in _to_iterable(mapper.map(icd_code, source="icd10", target="ccsr")))
                ccs_codes.update(code for code in mapped_ccs if code in ccs_code_to_desc)
                ccsr_codes.update(code for code in mapped_ccsr if code in ccsr_code_to_desc)

    def _format_icd_section(label: str, codes: List[str]) -> Optional[str]:
        if not codes:
            return None
        extra = icd_extra_counts.get(label, 0)
        body = ", ".join(codes)
        suffix = f" (+{extra} more)" if extra else ""
        return f"{label} codes: [{body}]{suffix}"

    parts: List[str] = []
    for label in ("ICD-9", "ICD-10", "ICD"):
        section = _format_icd_section(label, icd_codes_by_version.get(label, []))
        if section:
            parts.append(section)

    if ccs_codes:
        entries = [f"{code} {ccs_code_to_desc[code]}" for code in sorted(ccs_codes)]
        parts.append("CCS categories: [" + "; ".join(entries) + "]")

    if ccsr_codes:
        entries = [f"{code} {ccsr_code_to_desc[code]}" for code in sorted(ccsr_codes)]
        parts.append("CCSR categories: [" + "; ".join(entries) + "]")

    return " ; ".join(parts)


def _render_cxr(studies: List[dict]) -> str:
    if not studies:
        return ""
    lines = []
    for study in studies[:10]:
        study_id = study.get("study_id")
        images = study.get("images") or []
        first_time = None
        view_positions = set()
        for img in images:
            if first_time is None and img.get("study_datetime"):
                first_time = img["study_datetime"]
            view = img.get("view_position")
            if view and str(view).strip():
                view_positions.add(str(view).strip())
        header = f"Study {study_id}"
        if first_time:
            header += f" at {first_time}"
        header += f" with {len(images)} image(s)"
        if view_positions:
            header += f"; views: {', '.join(sorted(view_positions))}"
        block_lines = [header]
        report_text = study.get("report_text")
        if report_text:
            block_lines.append("Report:\n" + str(report_text).strip())
        lines.append("\n".join(block_lines))
    if len(studies) > 10:
        lines.append(f"... and {len(studies) - 10} additional studies omitted.")
    return "CXR imaging overview:\n" + "\n".join(lines)


def _render_ecg(records: List[dict]) -> str:
    if not records:
        return ""
    chunks = []
    for rec in records[:15]:
        study_id = rec.get("study_id")
        ecg_time = rec.get("ecg_time") or rec.get("ecg_time_matched")
        report = rec.get("report")
        parts = []
        header = f"ECG study {study_id}"
        if ecg_time:
            header += f" at {ecg_time}"
        parts.append(header + ".")
        if report:
            parts.append(report)
        else:
            axes = []
            for axis_key in ("rr_interval", "p_axis", "qrs_axis", "t_axis"):
                val = rec.get(axis_key)
                if val is not None:
                    axes.append(f"{axis_key}={val}")
            if axes:
                parts.append("Measurements: " + ", ".join(axes))
        chunks.append("\n".join(parts))
    if len(records) > 15:
        chunks.append(f"... and {len(records) - 15} additional ECG records omitted.")
    return "\n\n".join(chunks)


def _render_radiology_notes(notes: List[dict]) -> str:
    if not notes:
        return ""
    chunks = []
    for note in notes[:10]:
        parts = []
        charttime = note.get("charttime") or note.get("storetime")
        if charttime:
            parts.append(f"Reported at {charttime}.")
        matched = note.get("matched_cxr_study_id")
        if matched:
            parts.append(f"Linked CXR study ID: {matched}.")
        preview = note.get("text_preview")
        if preview:
            parts.append(preview)
        else:
            parts.append("(No radiology text available.)")
        chunks.append("\n".join(parts))
    if len(notes) > 10:
        chunks.append(f"... and {len(notes) - 10} additional radiology notes omitted.")
    return "\n\n".join(chunks)


def _render_discharge_note(note: Optional[dict], max_chars: int) -> str:
    if not note:
        return ""
    text = note.get("text")
    if not text:
        return ""
    return _truncate_text(text, max_chars)


def record_to_visit(record: dict, truncate_chars: int) -> VisitNarrative:
    modalities: List[ModalityText] = []

    if record.get("has_ehr") and record.get("diagnoses"):
        diag_text = _render_diagnoses(record["diagnoses"])
        if diag_text:
            modalities.append(ModalityText("ehr", diag_text))

    if record.get("has_cxr") and record.get("cxr_studies"):
        cxr_text = _render_cxr(record["cxr_studies"])
        if cxr_text:
            modalities.append(ModalityText("cxr", cxr_text))

    if record.get("has_ecg") and record.get("ecg_records"):
        ecg_text = _render_ecg(record["ecg_records"])
        if ecg_text:
            modalities.append(ModalityText("ecg", ecg_text))

    if record.get("has_discharge_note") and record.get("discharge_note"):
        discharge_text = _render_discharge_note(record["discharge_note"], truncate_chars)
        if discharge_text:
            modalities.append(ModalityText("discharge_note", discharge_text))

    if record.get("has_radiology_note") and record.get("radiology_notes"):
        rad_text = _render_radiology_notes(record["radiology_notes"])
        if rad_text:
            modalities.append(ModalityText("radiology_note", rad_text))

    return VisitNarrative(
        visit_id=str(record.get("hadm_id")),
        subject_id=int(record.get("subject_id")) if record.get("subject_id") is not None else None,
        is_virtual=bool(record.get("is_virtual", False)),
        start_time=_safe_iso(record.get("admittime")),
        end_time=_safe_iso(record.get("dischtime")),
        modalities=modalities,
    )


def load_visits_from_pickle(
    pickle_path: Path,
    subject_id: Optional[int] = None,
    max_visits: Optional[int] = None,
    truncate_chars: int = 2000,
    min_modalities: int = 1,
) -> Tuple[List[VisitNarrative], int]:
    import pickle

    records = pickle.load(pickle_path.open("rb"))

    grouped = defaultdict(list)
    for record in records:
        grouped[int(record["subject_id"])].append(record)

    def modality_count(rec: dict) -> int:
        flags = [
            rec.get("has_ehr", False),
            rec.get("has_cxr", False),
            rec.get("has_ecg", False),
            rec.get("has_discharge_note", False),
            rec.get("has_radiology_note", False),
        ]
        return sum(bool(flag) for flag in flags)

    selected_subject = None
    selected_pid = None
    if subject_id is not None:
        selected_subject = grouped.get(int(subject_id))
        if not selected_subject:
            raise ValueError(f"Subject {subject_id} not found in {pickle_path}.")
        selected_pid = int(subject_id)
    else:
        best_choice = None
        best_score = (-1, -1)
        for pid, recs in grouped.items():
            counts = [modality_count(r) for r in recs]
            if not counts:
                continue
            top_mod = max(counts)
            if top_mod < min_modalities:
                continue
            score = (top_mod, len(recs))
            if score > best_score:
                best_score = score
                best_choice = (int(pid), recs)
        if not best_choice:
            raise ValueError("No subject meeting modality criteria found in dataset.")
        selected_pid, selected_subject = best_choice

    sorted_records = sorted(selected_subject, key=_sort_key_for_record)
    visits = [
        record_to_visit(record, truncate_chars=truncate_chars)
        for record in sorted_records
    ]

    if max_visits is not None and max_visits > 0:
        visits = visits[:max_visits]

    return visits, selected_pid


# ---------------------------------------------------------------------------
# Dummy data
# ---------------------------------------------------------------------------


def _dummy_patient_sequence() -> List[VisitNarrative]:
    """Produce a tiny example patient with three visits."""

    return [
        VisitNarrative(
            visit_id="HADM123456",
            is_virtual=False,
            start_time="2023-01-05",
            end_time="2023-01-12",
            modalities=[
                ModalityText(
                    "ehr",
                    "CCSR: CIR007 (Congestive heart failure) - acute decompensation with "
                    "fluid overload; medications adjusted (furosemide up-titrated).",
                ),
                ModalityText(
                    "discharge_note",
                    "Patient stabilized after IV diuretics. Discharge plan includes "
                    "sodium restriction, daily weights, and close cardiology follow-up.",
                ),
                ModalityText(
                    "ecg",
                    "ECG 2023-01-06: sinus tachycardia, possible left ventricular hypertrophy, "
                    "QTc 472 ms.",
                ),
                ModalityText(
                    "radiology_note",
                    "Chest X-ray impression: pulmonary vascular congestion, trace pleural effusions.",
                ),
            ],
        ),
        VisitNarrative(
            visit_id="HADM123987",
            is_virtual=False,
            start_time="2023-04-18",
            end_time="2023-04-21",
            modalities=[
                ModalityText(
                    "ehr",
                    "CCSR: RSP008 (Chronic obstructive pulmonary disease) - mild exacerbation. "
                    "CCSR: CIRC005 (Essential hypertension) - controlled.",
                ),
                ModalityText(
                    "ecg",
                    "ECG 2023-04-19: normal sinus rhythm, no acute ischemic changes.",
                ),
                ModalityText(
                    "discharge_note",
                    "COPD flare managed with nebulized bronchodilators and short steroid taper. "
                    "Patient educated on inhaler technique.",
                ),
            ],
        ),
        VisitNarrative(
            visit_id="VIRTUAL-POST-1",
            is_virtual=True,
            start_time="2023-07-10",
            end_time="2023-07-10",
            modalities=[
                ModalityText(
                    "cxr",
                    "Outpatient CXR: lungs clear; interval resolution of prior effusions.",
                ),
                ModalityText(
                    "radiology_note",
                    "Impression: no acute cardiopulmonary process. Chronic cardiomegaly unchanged.",
                ),
                ModalityText(
                    "ecg",
                    "ECG 2023-07-10: sinus bradycardia, QTc 460 ms, otherwise unremarkable.",
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Summarization logic
# ---------------------------------------------------------------------------


SUMMARY_INSTRUCTIONS = """\
You are generating a longitudinal clinical summary using a sliding-window approach. This summary will serve as a concise text modality for multimodal generation tasks. You will receive the previous visit's summary and current visit data organized by modality.

**CRITICAL PRIORITIES:**
1. **Focus on disease-specific modalities**: Prioritize EHR diagnoses, CXR imaging findings, and ECG results as they directly reflect disease condition and clinical changes. Discharge notes and radiology reports are more general and should be used only to supplement the primary clinical findings.
2. **Track disease progression**: Your primary task is to identify and clearly articulate changes in the patient's condition between visits—what has improved, worsened, or remained stable.
3. **Distinguish acute vs chronic conditions**: Explicitly differentiate between acute disease events (e.g., "acute CHF exacerbation", "new pneumonia") and chronic/long-term conditions (e.g., "chronic COPD", "longstanding hypertension"). This distinction is critical for understanding disease evolution.
4. **Medical accuracy**: Stay strictly aligned with the provided clinical data. Do not infer or speculate beyond what is documented. If findings are ambiguous, acknowledge the uncertainty.
5. **Conciseness**: Avoid repeating static demographic information (age, gender, etc.) or unchanged clinical details unless they are essential for understanding disease progression.

**Output Format:**
Respond with a single valid JSON object containing these keys:
- "clinical_summary": A concise synthesis of the current visit focusing on active disease processes and acute findings from EHR, CXR, and ECG data. Clearly distinguish between acute events and chronic baseline conditions.
- "disease_progression": Explicit comparison with the prior visit—describe what changed (improved, worsened, new findings, resolved issues). Focus on both acute disease evolution and changes in chronic condition management. If no prior visit exists or no changes occurred, state that clearly.

**Content Guidelines:**
- Synthesize information across modalities, but give EHR diagnoses, CXR findings, and ECG interpretations the highest weight
- Use temporal language: "Since last visit...", "New finding:", "Resolved:", "Stable:", "Worsened:"
- Omit repetitive documentation of unchanged baseline conditions unless they contextualize new findings
- If data for a field is unavailable, set it to null
- Write in clear medical prose (no bullet points or markdown formatting within JSON values)
- Keep each field concise—aim for 2-4 sentences per field

**Example reasoning:** If the patient had acute CHF exacerbation last visit and now shows clear lungs on CXR with stable ECG, state: "Acute CHF exacerbation resolved, lungs now clear on CXR. Chronic CHF remains, currently compensated." Rather than re-listing demographic details or repeating unchanged chronic conditions without context.
"""


def build_visit_user_content(
    visit: VisitNarrative,
    prior_summary: Optional[str],
    instructions: str = SUMMARY_INSTRUCTIONS,
) -> str:
    sections: List[str] = []
    sections.append(instructions.strip())
    if prior_summary:
        sections.append("Prior summary:\n" + prior_summary.strip())
    sections.append("Source notes:\n" + visit.format_for_prompt())
    sections.append(
        "Task: Write the updated summary for this visit. "
        "If the patient status is unchanged, state that explicitly."
    )
    return "\n\n".join(sections)


def format_summary_output(visit: VisitNarrative, summary: str, step: int) -> str:
    tag = "virtual visit" if visit.is_virtual else "admission"
    time_window = ""
    if visit.start_time or visit.end_time:
        time_window = f" | window: {visit.start_time or '?'} → {visit.end_time or '?'}"
    header = f"=== Visit {step}: {visit.visit_id} ({tag}){time_window} ==="
    body = summary.strip()
    return f"{header}\n{body}\n"


def extract_summary_json(summary: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = summary.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None, "No JSON object detected in model output."
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


def generate_visit_summaries(
    visits: Iterable[VisitNarrative],
    llm: LLM,
    sampling: SamplingParams,
    system_prompt: Optional[str] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> List[str]:
    """Run chained summarization over the provided visit sequence."""

    summaries: List[str] = []
    for idx, visit in enumerate(visits):
        prior = summaries[-1] if summaries else None
        user_content = build_visit_user_content(visit, prior_summary=prior)

        if tokenizer is not None:
            messages: List[dict] = []
            system_msg = system_prompt or "You are a clinical summarizer."
            messages.append({"role": "system", "content": system_msg.strip()})
            messages.append({"role": "user", "content": user_content})
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_parts = []
            if system_prompt:
                prompt_parts.append(system_prompt.strip())
            prompt_parts.append(user_content)
            prompt = "\n\n".join(prompt_parts)

        outputs = llm.generate(prompt, sampling, use_tqdm=False)
        request_output = outputs[0] if isinstance(outputs, list) else outputs
        summary = request_output.outputs[0].text.strip()
        summaries.append(summary)
        print(format_summary_output(visit, summary, idx + 1))
    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based longitudinal summarization demo.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path compatible with vLLM (default: %(default)s).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Optional quantization config (e.g., awq, fp8, compressed-tensors).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=800,
        help="Maximum number of new tokens to generate per visit summary.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use built-in dummy patient visits (default true if no input).",
    )
    parser.add_argument(
        "--pickle-path",
        type=Path,
        default=None,
        help="Path to matching_results.pkl (e.g., output/v1/matching_results.pkl) to build visits from.",
    )
    parser.add_argument(
        "--subject-id",
        type=int,
        default=None,
        help="Specific subject_id to visualize from the pickle dataset. If omitted, the script picks a candidate automatically.",
    )
    parser.add_argument(
        "--max-visits",
        type=int,
        default=3,
        help="Maximum number of visits to include from the selected sequence.",
    )
    parser.add_argument(
        "--note-truncate",
        type=int,
        default=2000,
        help="Maximum number of characters to keep per long-form note (0 disables truncation).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the generated visit summaries as JSON.",
    )
    parser.add_argument(
        "--visits-json",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSON file containing visit sequences. "
            "File must store a list of visits with the same schema as VisitNarrative."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are an expert clinical summarizer.",
        help="System prompt injected ahead of user content (used with chat template).",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Format prompts with the model tokenizer's chat template (recommended for instruction-tuned models).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Whether to set trust_remote_code=True when loading the tokenizer.",
    )
    return parser.parse_args()


def load_visits_from_json(path: Path) -> List[VisitNarrative]:
    """Read a visit sequence exported as JSON."""
    raw = json.loads(path.read_text())
    visits: List[VisitNarrative] = []
    for item in raw:
        modalities = [
            ModalityText(modality=m["modality"], text=m["text"])
            for m in item.get("modalities", [])
        ]
        visits.append(
            VisitNarrative(
                visit_id=item["visit_id"],
                is_virtual=item.get("is_virtual", False),
                start_time=item.get("start_time"),
                end_time=item.get("end_time"),
                modalities=modalities,
            )
        )
    return visits


def main() -> None:
    args = parse_args()

    if args.pickle_path:
        visits, selected_subject = load_visits_from_pickle(
            args.pickle_path,
            subject_id=args.subject_id,
            max_visits=args.max_visits,
            truncate_chars=args.note_truncate,
        )
    elif args.visits_json:
        visits = load_visits_from_json(args.visits_json)
        selected_subject = None
    else:
        visits = _dummy_patient_sequence()
        selected_subject = None

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
    )

    tokenizer = None
    if args.use_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_output_tokens,
    )

    summaries = generate_visit_summaries(
        visits=visits,
        llm=llm,
        sampling=sampling,
        system_prompt=args.system_prompt,
        tokenizer=tokenizer,
    )

    if selected_subject is not None:
        print(f"Summaries generated for subject_id={selected_subject}.")

    export_payload = []
    for idx, (visit, summary) in enumerate(zip(visits, summaries), 1):
        parsed_json, parse_error = extract_summary_json(summary)
        export_payload.append(
            {
                "visit_index": idx,
                "subject_id": visit.subject_id,
                "hadm_id": visit.visit_id,
                "is_virtual": visit.is_virtual,
                "admittime": visit.start_time,
                "dischtime": visit.end_time,
                "modalities_present": [mt.modality for mt in visit.modalities],
                "modalities_text": [
                    {"modality": mt.modality, "text": mt.text}
                    for mt in visit.modalities
                ],
                "raw_summary": summary,
                "summary_json": parsed_json,
                "parse_error": parse_error,
            }
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        print(f"Saved summaries to {args.output_json}")


if __name__ == "__main__":
    main()


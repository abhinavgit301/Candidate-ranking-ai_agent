import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y", "%Y-%m", "%Y"]


def load_candidates(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    candidates: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line in tqdm(input_file, desc="Loading candidates", unit="record"):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue
            candidates.append(candidate)

    return candidates


def _ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if item is not None)
    return str(value).strip()


def _normalize_duration(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (int, float)):
        return str(int(value))
    duration_text = str(value).strip()
    if not duration_text:
        return "unknown"
    return duration_text


def _parse_date(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    date_text = str(value).strip()
    if not date_text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(date_text)
    except ValueError:
        return None


def get_recent_roles(career_history: Any) -> List[Dict[str, Any]]:
    if not isinstance(career_history, list):
        return []

    valid_roles = [role for role in career_history if isinstance(role, dict)]

    def sort_key(role: Dict[str, Any]) -> Tuple[int, Optional[datetime], Optional[datetime]]:
        start_date = _parse_date(role.get("start_date"))
        end_date = _parse_date(role.get("end_date"))
        date_key = start_date or end_date
        has_date = 1 if date_key is not None else 0
        return (has_date, date_key or datetime.min, end_date or datetime.min)

    sorted_roles = sorted(valid_roles, key=sort_key, reverse=True)
    return sorted_roles[:4] if len(sorted_roles) > 4 else sorted_roles


def _get_parquet_engine() -> Optional[str]:
    try:
        import pyarrow  # type: ignore

        return "pyarrow"
    except ImportError:
        try:
            import fastparquet  # type: ignore

            return "fastparquet"
        except ImportError:
            return None


def build_profile_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    current_title = _ensure_text(profile.get("current_title", ""))
    headline = _ensure_text(profile.get("headline", ""))
    summary = _ensure_text(profile.get("summary", ""))
    years_experience = _ensure_text(profile.get("years_of_experience", ""))
    current_company = _ensure_text(profile.get("current_company", ""))
    current_industry = _ensure_text(profile.get("current_industry", ""))

    lines = [
        f"CURRENT TITLE: {current_title}",
        f"HEADLINE: {headline}",
        f"YEARS OF EXPERIENCE: {years_experience}",
        f"CURRENT COMPANY: {current_company}",
        f"CURRENT INDUSTRY: {current_industry}",
        "",
        "SUMMARY:",
        summary,
    ]
    profile_text = "\n".join(lines).strip()
    if not any([current_title, headline, summary, years_experience, current_company, current_industry]):
        return ""
    return profile_text


def build_career_text(candidate: Dict[str, Any]) -> str:
    career_history = candidate.get("career_history", [])
    recent_roles = get_recent_roles(career_history)

    if not recent_roles:
        return ""

    blocks: List[str] = ["CAREER HISTORY:"]
    for job in recent_roles:
        title = _ensure_text(job.get("title", ""))
        company = _ensure_text(job.get("company", ""))
        industry = _ensure_text(job.get("industry", ""))
        company_size = _ensure_text(job.get("company_size", ""))
        duration = _normalize_duration(job.get("duration_months"))
        description = _ensure_text(job.get("description", ""))

        blocks.extend(
            [
                "ROLE TITLE:",
                title,
                "ROLE COMPANY:",
                company,
                "ROLE INDUSTRY:",
                industry,
                "ROLE COMPANY SIZE:",
                company_size,
                "ROLE DURATION:",
                f"{duration} months" if duration != "unknown" else "unknown",
                "ROLE DESCRIPTION:",
                description,
                "",
            ]
        )

    return "\n".join(blocks).strip()


def build_skills_text(candidate: Dict[str, Any]) -> str:
    skills = candidate.get("skills", [])
    if not isinstance(skills, list):
        skills = []

    entries: List[str] = []
    for skill in skills:
        if isinstance(skill, dict):
            name = _ensure_text(skill.get("name", ""))
            proficiency = _ensure_text(skill.get("proficiency", ""))
            duration = _normalize_duration(skill.get("duration_months"))
            details: List[str] = []
            if proficiency:
                details.append(proficiency.capitalize())
            if duration != "unknown":
                details.append(f"{duration} months")
            if name:
                if details:
                    entries.append(f"{name} ({', '.join(details)})")
                else:
                    entries.append(name)
        else:
            skill_text = _ensure_text(skill)
            if skill_text:
                entries.append(skill_text)

    if not entries:
        return ""

    return "\n".join(["SKILLS:", *entries]).strip()


def build_education_text(candidate: Dict[str, Any]) -> str:
    education_history = candidate.get("education", [])
    if not isinstance(education_history, list):
        education_history = []

    blocks: List[str] = []
    for education in education_history:
        if not isinstance(education, dict):
            continue

        institution = _ensure_text(education.get("institution", ""))
        degree = _ensure_text(education.get("degree", ""))
        field_of_study = _ensure_text(education.get("field_of_study", ""))
        tier = _ensure_text(education.get("tier", ""))

        if not any([institution, degree, field_of_study, tier]):
            continue

        blocks.extend(
            [
                f"Institution: {institution}",
                f"Degree: {degree}",
                f"Field: {field_of_study}",
                f"Tier: {tier}",
                "",
            ]
        )

    if not blocks:
        return ""

    return "\n".join(["EDUCATION:", *blocks]).strip()


def build_certifications_text(candidate: Dict[str, Any]) -> str:
    certifications = candidate.get("certifications", [])
    if not isinstance(certifications, list):
        certifications = []

    entries: List[str] = []
    for certification in certifications:
        if isinstance(certification, dict):
            name = _ensure_text(certification.get("name", ""))
            if name:
                entries.append(name)
        else:
            cert_text = _ensure_text(certification)
            if cert_text:
                entries.append(cert_text)

    if not entries:
        return ""

    return "\n".join(["CERTIFICATIONS:", *entries]).strip()


def _normalize_keyword(value: str) -> str:
    return value.strip().lower().replace(" ", " ")


def _normalize_token(token: str) -> str:
    return token.lower().strip()


TECHNOLOGY_KEYWORDS: Dict[str, str] = {
    "python": "Python",
    "sql": "SQL",
    "spark": "Spark",
    "hive": "Hive",
    "kafka": "Kafka",
    "airflow": "Airflow",
    "snowflake": "Snowflake",
    "dbt": "dbt",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit-learn": "Scikit-learn",
    "mlflow": "MLflow",
    "hadoop": "Hadoop",
    "elasticsearch": "Elasticsearch",
    "redis": "Redis",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "bigquery": "BigQuery",
    "tableau": "Tableau",
    "mssql": "SQL Server",
    "sql server": "SQL Server",
    "spark sql": "Spark SQL",
    "feature engineering": "Feature Engineering",
    "ci/cd": "CI/CD",
    "api": "APIs",
    "rest": "REST",
}

AI_SIGNAL_KEYWORDS: Dict[str, str] = {
    "nlp": "NLP",
    "natural language processing": "NLP",
    "llm": "LLM Fine-tuning",
    "fine-tuning": "LLM Fine-tuning",
    "lora": "LoRA",
    "vector database": "Vector Databases",
    "vector databases": "Vector Databases",
    "faiss": "Vector Databases",
    "milvus": "Vector Databases",
    "pinecone": "Vector Databases",
    "speech recognition": "Speech Recognition",
    "computer vision": "Computer Vision",
    "model training": "Model Training",
    "training": "Model Training",
    "feature engineering": "Feature Engineering",
    "mlops": "MLOps",
    "data science": "Data Science",
    "recommendation": "Recommendation Systems",
    "recommendation systems": "Recommendation Systems",
}

ROLE_SPECIALIZATION_PATTERNS: List[Tuple[str, List[str]]] = [
    ("Data Engineer", ["data engineer", "etl engineer", "pipeline engineer", "data platform"]),
    ("Backend Engineer", ["backend engineer", "backend developer", "server engineer"]),
    ("Analytics Engineer", ["analytics engineer", "analytics developer"]),
    ("Machine Learning Engineer", ["machine learning engineer", "ml engineer", "mlops engineer"]),
    ("Data Scientist", ["data scientist", "ml scientist"]),
    ("Data Analyst", ["data analyst", "analytics analyst"]),
    ("Product Manager", ["product manager", "product owner"]),
    ("Marketing Manager", ["marketing manager"]),
    ("Operations Manager", ["operations manager", "ops manager"]),
    ("DevOps Engineer", ["devops engineer", "site reliability engineer", "sre"]),
    ("Software Engineer", ["software engineer", "software developer", "developer"]),
]

DOMAIN_PATTERNS: List[Tuple[str, List[str]]] = [
    ("Data Engineering", ["data engineer", "etl", "data pipelines", "data platform", "spark", "airflow", "snowflake", "dbt", "bigquery", "hive"]),
    ("Backend Engineering", ["backend engineer", "backend developer", "rest api", "microservices", "server engineer", "java", "python"]),
    ("Analytics", ["analytics engineer", "analytics", "tableau", "power bi", "lookml", "business intelligence", "bi"]),
    ("Machine Learning", ["machine learning", "ml", "mlops", "data science", "model training", "predictive"]),
]

RETRIEVAL_SIGNAL_KEYWORDS: Dict[str, str] = {
    "data pipeline": "Data Pipelines",
    "data pipelines": "Data Pipelines",
    "machine learning": "Machine Learning",
    "mlops": "MLOps",
    "real-time": "Real-Time",
    "real time": "Real-Time",
    "data engineering": "Data Engineering",
    "backend engineering": "Backend Engineering",
    "analytics": "Analytics",
    "artificial intelligence": "AI",
    "ai": "AI",
    "deep learning": "Deep Learning",
    "computer vision": "Computer Vision",
    "natural language processing": "NLP",
}


def _extract_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9+#]+(?:\s+[A-Za-z0-9+#]+)*", text)


def _extract_concepts(text: str, mapping: Dict[str, str]) -> List[str]:
    normalized = text.lower()
    found: List[str] = []
    for key, canonical in mapping.items():
        if key in normalized and canonical not in found:
            found.append(canonical)
    return found


def _unique_ordered(items: List[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def get_candidate_skill_entries(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    skills = candidate.get("skills", [])
    if isinstance(skills, list):
        for skill in skills:
            if isinstance(skill, dict):
                name = _ensure_text(skill.get("name", ""))
                proficiency = _ensure_text(skill.get("proficiency", ""))
                duration = _normalize_duration(skill.get("duration_months"))
                if name:
                    entries.append({"name": name, "proficiency": proficiency, "duration": duration})
            else:
                name = _ensure_text(skill)
                if name:
                    entries.append({"name": name, "proficiency": "", "duration": "unknown"})
    return entries


def get_candidate_skill_names(candidate: Dict[str, Any]) -> List[str]:
    entries = get_candidate_skill_entries(candidate)
    names = [entry["name"] for entry in entries]
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    title = _ensure_text(profile.get("current_title", ""))
    summary = _ensure_text(profile.get("summary", ""))
    career_history = candidate.get("career_history", [])
    text_sources = [title, summary]
    if isinstance(career_history, list):
        for job in career_history:
            if isinstance(job, dict):
                text_sources.append(_ensure_text(job.get("title", "")))
                text_sources.append(_ensure_text(job.get("description", "")))
    inferred = _extract_concepts(" ".join(token for token in text_sources if token), {**TECHNOLOGY_KEYWORDS, **AI_SIGNAL_KEYWORDS})
    return _unique_ordered([*names, *inferred])


def get_primary_secondary_skills(candidate: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    entries = get_candidate_skill_entries(candidate)
    inferred = get_candidate_skill_names(candidate)
    primary_candidates: List[str] = []
    secondary_candidates: List[str] = []
    for entry in entries:
        name = entry["name"]
        proficiency = entry["proficiency"].lower()
        duration = entry["duration"]
        months = int(duration) if isinstance(duration, str) and duration.isdigit() else 0
        if proficiency in {"advanced", "expert", "proficient"} or months >= 18:
            primary_candidates.append(name)
        else:
            secondary_candidates.append(name)

    primary = _unique_ordered(primary_candidates + [skill for skill in inferred if skill not in primary_candidates])[:10]
    secondary = _unique_ordered([skill for skill in inferred if skill not in primary][:10] + [skill for skill in secondary_candidates if skill not in primary])[:10]
    return primary, secondary


def build_primary_skills_text(candidate: Dict[str, Any]) -> str:
    primary_skills, _ = get_primary_secondary_skills(candidate)
    if not primary_skills:
        return ""
    return "\n".join(["PRIMARY SKILLS:", *primary_skills])


def build_secondary_skills_text(candidate: Dict[str, Any]) -> str:
    _, secondary_skills = get_primary_secondary_skills(candidate)
    if not secondary_skills:
        return ""
    return "\n".join(["SECONDARY SKILLS:", *secondary_skills])


def get_domains(candidate: Dict[str, Any]) -> List[str]:
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    text_sources: List[str] = [
        _ensure_text(profile.get("summary", "")),
        _ensure_text(profile.get("current_title", "")),
        _ensure_text(profile.get("current_industry", "")),
    ]
    career_history = candidate.get("career_history", [])
    if isinstance(career_history, list):
        for role in career_history:
            if isinstance(role, dict):
                text_sources.append(_ensure_text(role.get("title", "")))
                text_sources.append(_ensure_text(role.get("description", "")))

    normalized = " ".join(token.lower() for token in text_sources if token)
    domains: List[str] = []
    for domain, patterns in DOMAIN_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            domains.append(domain)
    return _unique_ordered(domains)


def infer_career_focus(candidate: Dict[str, Any], specialization: str, domains: List[str]) -> str:
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    current_industry = _ensure_text(profile.get("current_industry", ""))
    if specialization:
        return specialization
    if domains:
        return domains[0]
    return current_industry


def _experience_level(years: Any) -> str:
    try:
        value = float(years)
    except (TypeError, ValueError):
        return ""
    if value < 3:
        return "Junior"
    if value < 7:
        return "Mid"
    return "Senior"


def estimate_ai_ml_exposure(candidate: Dict[str, Any]) -> str:
    signals = get_ai_ml_signals(candidate)
    if len(signals) >= 4:
        return "High"
    if len(signals) >= 1:
        return "Moderate"
    return "Low"


def _has_leadership_experience(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    text_parts = [
        _ensure_text(profile.get("summary", "")),
        _ensure_text(profile.get("current_title", "")),
    ]
    career_history = candidate.get("career_history", [])
    if isinstance(career_history, list):
        for role in career_history:
            if isinstance(role, dict):
                text_parts.append(_ensure_text(role.get("title", "")))
                text_parts.append(_ensure_text(role.get("description", "")))

    normalized = " ".join(token.lower() for token in text_parts if token)
    leadership_terms = ["lead", "leadership", "manager", "management", "supervisor", "director", "head", "architect", "principal", "owner"]
    return "Yes" if any(term in normalized for term in leadership_terms) else "No"


def build_candidate_summary(candidate: Dict[str, Any], specialization: str, domains: List[str], ai_ml_exposure: str, leadership_experience: str) -> str:
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    experience_level = _experience_level(profile.get("years_of_experience", ""))
    years_experience = _ensure_text(profile.get("years_of_experience", ""))
    current_title = _ensure_text(profile.get("current_title", ""))
    current_industry = _ensure_text(profile.get("current_industry", ""))
    career_focus = infer_career_focus(candidate, specialization, domains)

    lines = [
        "CANDIDATE SUMMARY:",
        f"Experience Level: {experience_level or 'Unknown'}",
        f"Years of Experience: {years_experience or 'Unknown'}",
        f"Current Role: {current_title or 'Unknown'}",
        f"Current Industry: {current_industry or 'Unknown'}",
        f"Career Focus: {career_focus or 'Unknown'}",
        f"AI/ML Exposure: {ai_ml_exposure}",
        f"Leadership Experience: {leadership_experience}",
    ]
    return "\n".join(lines)


def build_domains_text(domains: List[str]) -> str:
    if not domains:
        return ""
    return "\n".join(["DOMAINS:", *domains])


def build_career_trajectory(candidate: Dict[str, Any]) -> str:
    career_history = candidate.get("career_history", [])
    if not isinstance(career_history, list):
        return ""

    recent_roles = get_recent_roles(career_history)
    titles: List[str] = []
    for role in recent_roles:
        title = _ensure_text(role.get("title", ""))
        if title and title not in titles:
            titles.append(title)

    if not titles:
        return ""

    return "\n".join(["CAREER TRAJECTORY:", *titles[:5]])


def _rank_concepts(text: str, mapping: Dict[str, str], limit: int = 15) -> List[str]:
    normalized = text.lower()
    counts: Counter = Counter()
    for key, canonical in mapping.items():
        if key in normalized:
            counts[canonical] += normalized.count(key)

    result: List[str] = []
    for concept, _ in counts.most_common(limit):
        if concept not in result:
            result.append(concept)
    return result


def build_key_retrieval_signals(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    summary = _ensure_text(profile.get("summary", ""))
    title = _ensure_text(profile.get("current_title", ""))
    career_history = candidate.get("career_history", [])
    skills = get_candidate_skill_names(candidate)

    text_sources: List[str] = [summary, title, *skills]
    if isinstance(career_history, list):
        for role in career_history:
            if isinstance(role, dict):
                text_sources.append(_ensure_text(role.get("title", "")))
                text_sources.append(_ensure_text(role.get("description", "")))

    combined = " ".join(token for token in text_sources if token)
    signals = _rank_concepts(combined, {**TECHNOLOGY_KEYWORDS, **AI_SIGNAL_KEYWORDS, **RETRIEVAL_SIGNAL_KEYWORDS}, 15)
    if not signals:
        return ""

    return "\n".join(["KEY RETRIEVAL SIGNALS:", *signals])


def build_primary_career_signals(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    career_history = candidate.get("career_history", [])
    if not isinstance(career_history, list):
        career_history = []

    signals: List[str] = []
    current_title = _ensure_text(profile.get("current_title", ""))
    if current_title:
        signals.append(current_title)

    recent_roles = get_recent_roles(career_history)
    for role in recent_roles:
        title = _ensure_text(role.get("title", ""))
        if title and title not in signals:
            signals.append(title)

    description_text = " ".join(_ensure_text(role.get("description", "")) for role in recent_roles)
    keywords = re.findall(r"\b[A-Za-z][A-Za-z0-9&+-]{2,}\b", description_text)
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "have", "has",
        "were", "been", "data", "systems", "system", "using", "worked",
        "work", "team", "requirements", "project", "projects",
    }
    normalized: List[str] = []
    for token in keywords:
        lower = token.lower()
        if lower in stopwords:
            continue
        if token.isdigit() or len(lower) < 3:
            continue
        normalized.append(token)

    counts = Counter(normalized)
    for keyword, _count in counts.most_common(6):
        if keyword not in signals:
            signals.append(keyword)

    if not signals:
        return ""

    return "\n".join(["PRIMARY CAREER SIGNALS:", *signals])


def infer_role_specialization(candidate: Dict[str, Any]) -> str:
    titles: List[str] = []
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    current_title = _ensure_text(profile.get("current_title", ""))
    if current_title:
        titles.append(current_title)

    career_history = candidate.get("career_history", [])
    if isinstance(career_history, list):
        recent_roles = get_recent_roles(career_history)
        for role in recent_roles:
            role_title = _ensure_text(role.get("title", ""))
            if role_title:
                titles.append(role_title)

    normalized_titles = [title.lower() for title in titles if title]
    for specialization, patterns in ROLE_SPECIALIZATION_PATTERNS:
        for pattern in patterns:
            if any(pattern in title for title in normalized_titles):
                return "\n".join(["ROLE SPECIALIZATION:", specialization])

    if current_title:
        return "\n".join(["ROLE SPECIALIZATION:", current_title])
    if titles:
        return "\n".join(["ROLE SPECIALIZATION:", titles[0]])
    return ""


def build_search_keywords(candidate: Dict[str, Any], tech_stack: List[str], ai_signals: List[str], specialization: str) -> str:
    keywords: List[str] = []
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    current_title = _ensure_text(profile.get("current_title", ""))
    if current_title:
        keywords.append(current_title.lower())
    if specialization:
        keywords.append(specialization.lower())

    skills = candidate.get("skills", [])
    if isinstance(skills, list):
        for skill in skills:
            if isinstance(skill, dict):
                name = _ensure_text(skill.get("name", ""))
                if name:
                    keywords.append(name.lower())
            else:
                skill_text = _ensure_text(skill)
                if skill_text:
                    keywords.append(skill_text.lower())

    for term in tech_stack + ai_signals:
        keywords.append(term.lower())

    career_history = candidate.get("career_history", [])
    if isinstance(career_history, list):
        for role in career_history:
            if isinstance(role, dict):
                title = _ensure_text(role.get("title", ""))
                if title:
                    keywords.append(title.lower())
                company = _ensure_text(role.get("company", ""))
                if company:
                    keywords.append(company.lower())

    entries: List[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        keyword = keyword.strip()
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)
        entries.append(keyword)
        if len(entries) >= 30:
            break

    if not entries:
        return ""

    keyword_lines = [k for k in entries if k]
    return "\n".join(["SEARCH KEYWORDS:", *keyword_lines])


def build_experience_summary(candidate: Dict[str, Any], tech_stack: List[str], specialization: str) -> str:
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    years_experience = _ensure_text(profile.get("years_of_experience", ""))
    current_company = _ensure_text(profile.get("current_company", ""))
    summary_lines: List[str] = ["EXPERIENCE SUMMARY:"]
    if years_experience:
        summary_lines.append(f"{years_experience} years experience.")
    if current_company:
        summary_lines.append(f"Worked at {current_company}.")
    if specialization:
        summary_lines.append(f"Focused on {specialization}.")
    tech_items = tech_stack[:3]
    for tech in tech_items:
        summary_lines.append(f"Experience with {tech}.")

    if len(summary_lines) <= 1:
        return ""

    return "\n".join(summary_lines)


def build_career_progression(candidate: Dict[str, Any]) -> str:
    career_history = candidate.get("career_history", [])
    if not isinstance(career_history, list):
        return ""

    recent_roles = get_recent_roles(career_history)
    titles: List[str] = []
    for role in reversed(recent_roles):
        title = _ensure_text(role.get("title", ""))
        if title and (not titles or title != titles[-1]):
            titles.append(title)

    if len(titles) < 2:
        return ""

    return "\n".join(["CAREER PROGRESSION:", " -> ".join(titles)])


def build_skill_assessment_text(candidate: Dict[str, Any]) -> str:
    signals = candidate.get("redrob_signals", {}) if isinstance(candidate, dict) else {}
    assessments = signals.get("skill_assessment_scores", {})
    if not isinstance(assessments, dict) or not assessments:
        return ""

    blocks: List[str] = ["SKILL ASSESSMENTS:"]
    for skill, score in assessments.items():
        skill_name = _ensure_text(skill)
        score_text = _ensure_text(score)
        if skill_name:
            blocks.append(f"{skill_name}: {score_text}")

    return "\n".join(blocks).strip()


def get_technology_stack(candidate: Dict[str, Any]) -> List[str]:
    text_sources: List[str] = []
    skills = candidate.get("skills", [])
    if isinstance(skills, list):
        for skill in skills:
            if isinstance(skill, dict):
                text_sources.append(_ensure_text(skill.get("name", "")))
            else:
                text_sources.append(_ensure_text(skill))
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    text_sources.append(_ensure_text(profile.get("summary", "")))
    career_history = candidate.get("career_history", [])
    if isinstance(career_history, list):
        for job in career_history:
            if isinstance(job, dict):
                text_sources.append(_ensure_text(job.get("description", "")))
                text_sources.append(_ensure_text(job.get("title", "")))

    combined = " ".join(token for token in text_sources if token)
    if not combined:
        return []

    techs = _extract_concepts(combined, TECHNOLOGY_KEYWORDS)
    return _unique_ordered(techs)


def build_technology_stack(candidate: Dict[str, Any]) -> str:
    techs = get_technology_stack(candidate)
    if not techs:
        return ""
    return "\n".join(["TECHNOLOGY STACK:", *techs])


def get_ai_ml_signals(candidate: Dict[str, Any]) -> List[str]:
    text_sources: List[str] = []
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    text_sources.append(_ensure_text(profile.get("summary", "")))
    career_history = candidate.get("career_history", [])
    if isinstance(career_history, list):
        for job in career_history:
            if isinstance(job, dict):
                text_sources.append(_ensure_text(job.get("description", "")))
                text_sources.append(_ensure_text(job.get("title", "")))
    skills = candidate.get("skills", [])
    if isinstance(skills, list):
        for skill in skills:
            if isinstance(skill, dict):
                text_sources.append(_ensure_text(skill.get("name", "")))
            else:
                text_sources.append(_ensure_text(skill))

    combined = " ".join(text_sources)
    return _unique_ordered(_extract_concepts(combined, AI_SIGNAL_KEYWORDS))


def build_ai_ml_signals(candidate: Dict[str, Any]) -> str:
    ai_signals = get_ai_ml_signals(candidate)
    if not ai_signals:
        return ""
    return "\n".join(["AI/ML SIGNALS:", *ai_signals])


def build_behavior_text(candidate: Dict[str, Any]) -> str:
    signals = candidate.get("redrob_signals", {}) if isinstance(candidate, dict) else {}
    open_to_work = _ensure_text(signals.get("open_to_work_flag", ""))
    last_active_date = _ensure_text(signals.get("last_active_date", ""))
    recruiter_response_rate = _ensure_text(signals.get("recruiter_response_rate", ""))
    interview_completion_rate = _ensure_text(signals.get("interview_completion_rate", ""))
    notice_period_days = _ensure_text(signals.get("notice_period_days", ""))

    if not any([open_to_work, last_active_date, recruiter_response_rate, interview_completion_rate, notice_period_days]):
        return ""

    lines = [
        "BEHAVIOR:",
        f"Open To Work: {open_to_work}",
        f"Last Active Date: {last_active_date}",
        f"Recruiter Response Rate: {recruiter_response_rate}",
        f"Interview Completion Rate: {interview_completion_rate}",
        f"Notice Period Days: {notice_period_days}",
    ]
    return "\n".join(lines).strip()


def build_retrieval_text(
    candidate: Dict[str, Any],
    specialization: str,
    tech_stack: List[str],
    ai_signals: List[str],
    keywords: List[str],
    retrieval_signals: List[str],
    primary_skills: List[str],
) -> str:
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    title = _ensure_text(profile.get("current_title", ""))
    skill_names = primary_skills[:10]

    parts: List[str] = []
    if title:
        parts.append(title)
    if specialization:
        parts.append(specialization)
    parts.extend(primary_skills)
    parts.extend(tech_stack)
    parts.extend(ai_signals)
    parts.extend(retrieval_signals)
    parts.extend(keywords)

    return " ".join(part for part in parts if part).strip()


def build_candidate_record(candidate: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    candidate_id = _ensure_text(candidate.get("candidate_id", ""))
    if not candidate_id:
        return None, "missing_candidate_id"

    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    if not isinstance(profile, dict) or not profile:
        return None, "missing_profile"

    signals = candidate.get("redrob_signals", {}) if isinstance(candidate, dict) else {}

    profile_text = build_profile_text(candidate)
    if not profile_text:
        return None, "missing_profile_text"

    specialization_text = infer_role_specialization(candidate)
    specialization = specialization_text.split("\n", 1)[1] if "\n" in specialization_text else specialization_text
    technology_stack_list = get_technology_stack(candidate)
    technology_stack_text = build_technology_stack(candidate)
    ai_ml_signals_list = get_ai_ml_signals(candidate)
    ai_ml_signals_text = build_ai_ml_signals(candidate)
    primary_skills_list, secondary_skills_list = get_primary_secondary_skills(candidate)
    primary_skills_text = build_primary_skills_text(candidate)
    secondary_skills_text = build_secondary_skills_text(candidate)
    domains_list = get_domains(candidate)
    domains_text = build_domains_text(domains_list)
    career_focus = infer_career_focus(candidate, specialization, domains_list)
    ai_ml_exposure = estimate_ai_ml_exposure(candidate)
    leadership_experience = _has_leadership_experience(candidate)
    candidate_summary_text = build_candidate_summary(candidate, specialization, domains_list, ai_ml_exposure, leadership_experience)
    career_trajectory_text = build_career_trajectory(candidate)
    key_retrieval_signals_text = build_key_retrieval_signals(candidate)
    keyword_text = build_search_keywords(candidate, technology_stack_list, ai_ml_signals_list, specialization)
    keyword_list = [line for line in keyword_text.split("\n")[1:]] if keyword_text else []
    experience_summary_text = build_experience_summary(candidate, technology_stack_list, specialization)
    career_progression_text = build_career_progression(candidate)
    career_text = build_career_text(candidate)
    skills_text = build_skills_text(candidate)
    skill_assessment_text = build_skill_assessment_text(candidate)
    primary_career_signals = build_primary_career_signals(candidate)
    education_text = build_education_text(candidate)
    certifications_text = build_certifications_text(candidate)
    behavior_text = build_behavior_text(candidate)

    full_text_parts = [
        candidate_summary_text,
        primary_skills_text,
        secondary_skills_text,
        domains_text,
        career_trajectory_text,
        key_retrieval_signals_text,
        profile_text,
        specialization_text,
        experience_summary_text,
        technology_stack_text,
        ai_ml_signals_text,
        keyword_text,
        career_progression_text,
        career_text,
        skills_text,
        skill_assessment_text,
        education_text,
        certifications_text,
        behavior_text,
    ]
    full_text = "\n".join([part for part in full_text_parts if part]).strip()

    if len(full_text) < 100:
        return None, "full_text_too_short"

    skill_count = 0
    skills = candidate.get("skills", [])
    if isinstance(skills, list):
        skill_count = len(skills)

    education_tier = None
    education_history = candidate.get("education", [])
    if isinstance(education_history, list) and education_history:
        first_edu = education_history[0]
        if isinstance(first_edu, dict):
            education_tier = first_edu.get("tier")

    ai_ml_score = len(ai_ml_signals_list) * 10

    metadata: Dict[str, Any] = {
        "years_of_experience": profile.get("years_of_experience"),
        "experience_level": _experience_level(profile.get("years_of_experience", "")),
        "current_title": profile.get("current_title"),
        "current_industry": profile.get("current_industry"),
        "open_to_work": signals.get("open_to_work_flag"),
        "notice_period_days": signals.get("notice_period_days"),
        "primary_skills": primary_skills_list,
        "domains": domains_list,
        "career_focus": career_focus,
        "ai_ml_exposure": ai_ml_exposure,
        "leadership_experience": leadership_experience,
        "current_company": profile.get("current_company"),
        "current_company_size": profile.get("current_company_size"),
        "specialization": specialization,
        "skills_count": skill_count,
        "education_tier": education_tier,
        "ai_ml_score": ai_ml_score,
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "tech_stack": technology_stack_list,
    }

    retrieval_signals = [line for line in key_retrieval_signals_text.split("\n")[1:]] if key_retrieval_signals_text else []
    retrieval_text = build_retrieval_text(candidate, specialization, technology_stack_list, ai_ml_signals_list, keyword_list, retrieval_signals, primary_skills_list)

    return {
        "candidate_id": candidate_id,
        "metadata": metadata,
        "candidate_summary_text": candidate_summary_text,
        "primary_skills_text": primary_skills_text,
        "secondary_skills_text": secondary_skills_text,
        "domains_text": domains_text,
        "career_trajectory_text": career_trajectory_text,
        "key_retrieval_signals_text": key_retrieval_signals_text,
        "profile_text": profile_text,
        "role_specialization_text": specialization_text,
        "experience_summary_text": experience_summary_text,
        "technology_stack_text": technology_stack_text,
        "ai_ml_signals_text": ai_ml_signals_text,
        "search_keywords_text": keyword_text,
        "career_progression_text": career_progression_text,
        "career_text": career_text,
        "skills_text": skills_text,
        "skill_assessment_text": skill_assessment_text,
        "primary_career_signals": primary_career_signals,
        "education_text": education_text,
        "certifications_text": certifications_text,
        "behavior_text": behavior_text,
        "retrieval_text": retrieval_text,
        "full_text": full_text,
    }, None


def save_candidate_texts(candidates: List[Dict[str, Any]], output_path: Path) -> Tuple[int, int, int, Dict[str, int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    records: List[Dict[str, Any]] = []
    skip_reasons: Dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as output_file:
        for candidate in tqdm(candidates, desc="Building candidate texts", unit="candidate"):
            candidate_id = _ensure_text(candidate.get("candidate_id", ""))
            if not candidate_id:
                skipped += 1
                skip_reasons["missing_candidate_id"] = skip_reasons.get("missing_candidate_id", 0) + 1
                continue

            try:
                record, reason = build_candidate_record(candidate)
            except Exception:
                skipped += 1
                skip_reasons["exception_building_record"] = skip_reasons.get("exception_building_record", 0) + 1
                continue

            if record is None:
                skipped += 1
                reason_key = reason or "validation_failure"
                skip_reasons[reason_key] = skip_reasons.get(reason_key, 0) + 1
                continue

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            processed += 1

    parquet_path = output_path.with_suffix(".parquet")
    dataframe = pd.DataFrame(records)
    engine = _get_parquet_engine()
    if engine:
        dataframe.to_parquet(parquet_path, compression="snappy", index=False, engine=engine)
    else:
        print(
            "Warning: parquet engine not installed. JSONL output was generated, but parquet output was skipped. "
            "Install pyarrow or fastparquet to enable parquet support."
        )
    return processed, skipped, len(records), skip_reasons


def print_debug_sample(candidates: List[Dict[str, Any]]) -> None:
    for candidate in candidates:
        record, _ = build_candidate_record(candidate)
        if record is not None:
            print("\nDEBUG SAMPLE RECORD:\n")
            print(json.dumps(record, indent=2, ensure_ascii=False))
            return
    print("No valid candidate sample available for debug.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate text records from Redrob data.")
    parser.add_argument("--debug", action="store_true", help="Print one sample candidate record after processing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "data" / "raw" / "candidates.jsonl"
    output_path = repo_root / "data" / "processed" / "candidate_texts.jsonl"

    candidates = load_candidates(input_path)
    processed, skipped, parquet_count, skip_reasons = save_candidate_texts(candidates, output_path)

    print(f"Total candidates processed: {processed}")
    print(f"Total candidates skipped: {skipped}")
    print(f"Output file location: {output_path}")

    if args.debug:
        jsonl_size = output_path.stat().st_size if output_path.exists() else 0
        parquet_path = output_path.with_suffix(".parquet")
        parquet_size = parquet_path.stat().st_size if parquet_path.exists() else 0
        print(f"JSONL size: {jsonl_size} bytes")
        print(f"Parquet size: {parquet_size} bytes")
        print(f"Parquet row count: {parquet_count}")
        if skip_reasons:
            print("Skipped record reasons:")
            for reason, count in skip_reasons.items():
                print(f"  {reason}: {count}")
        print_debug_sample(candidates)


if __name__ == "__main__":
    main()
